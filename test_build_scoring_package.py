"""Tests for build_scoring_package.py. Run with:
python3 -m unittest test_build_scoring_package.py"""

import json
import os
import tempfile
import unittest
import zipfile

from case_editor import CaseStore
from merge_cases import MasterStore
from eval_common import AnswersStore, case_hash
from build_scoring_package import (
    blind_cases,
    blinded_image_name,
    build_package,
    self_identification_warnings,
)


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.old_cwd = os.getcwd()
        os.chdir(self.dir)

        provider = CaseStore(3, "provider_003_cases.json")
        provider.add_case("Case one text.", ["r1", "r2"])
        provider.add_case("Case two text.", ["r1"])
        self.master = MasterStore("master_cases.json")
        self.master.merge_provider(provider)

        self.answers = AnswersStore.load_or_create("answers.json", "answer_images")
        os.makedirs("answer_images")
        self.image = os.path.join("answer_images", "003-001_siteb_001.png")
        self.page_image = os.path.join("answer_images", "003-001_siteb_page.png")
        for i, path in enumerate((self.image, self.page_image)):
            with open(path, "wb") as f:
                f.write(b"\x89PNG fake image %d" % i)

        for case_id in ("003-001", "003-002"):
            self.answers.upsert(
                {
                    "case_id": case_id,
                    "model_id": "modela",
                    "model_display_name": "Model A",
                    "model_reported": "modela-2026-01",
                    "status": "ok",
                    "response_text": "Answer from the first system for " + case_id,
                    "images": [],
                    "case_sha256": case_hash(self.master.cases[case_id]),
                }
            )
        self.answers.upsert(
            {
                "case_id": "003-001",
                "model_id": "siteb",
                "model_display_name": "Site B",
                "model_reported": "Site B (web interface, accessed 2026-07-12)",
                "status": "ok_manual",
                "response_text": "Answer with pictures for 003-001",
                "images": [self.image, self.page_image],
                "case_sha256": case_hash(self.master.cases["003-001"]),
            }
        )

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def build(self, name="testpkg", case_ids=None, model_ids=None):
        return build_package(
            name,
            case_ids or ["003-001", "003-002"],
            model_ids or ["modela", "siteb"],
            self.master,
            self.answers,
            warn=lambda *_: None,
        )

    def test_zip_and_key_are_consistent(self):
        zip_path, key_path = self.build()
        with zipfile.ZipFile(zip_path) as bundle:
            manifest = json.loads(bundle.read("package.json"))
        with open(key_path, encoding="utf-8") as f:
            key = json.load(f)

        self.assertEqual(manifest["package_id"], key["package_id"])
        self.assertEqual(manifest["num_cases"], 2)
        # Every label in the manifest must be resolvable through the key,
        # and the response text must match the model the key claims.
        for case in manifest["cases"]:
            for answer in case["answers"]:
                entry = key["key"][case["case_id"]][answer["label"]]
                record = self.answers.get(case["case_id"], entry["model_id"])
                self.assertEqual(answer["response_text"], record["response_text"])

    def test_no_model_names_leak_into_the_zip(self):
        zip_path, _ = self.build()
        with zipfile.ZipFile(zip_path) as bundle:
            blob = " ".join(bundle.namelist()).lower()
            for name in bundle.namelist():
                blob += bundle.read(name).decode("utf-8", "replace").lower()
        for secret in ("modela", "siteb", "model a", "site b"):
            self.assertNotIn(secret, blob)

    def test_images_are_renamed_to_blinded_labels(self):
        zip_path, key_path = self.build(case_ids=["003-001"])
        with zipfile.ZipFile(zip_path) as bundle:
            manifest = json.loads(bundle.read("package.json"))
            names = bundle.namelist()
        with open(key_path, encoding="utf-8") as f:
            key = json.load(f)["key"]

        label_for_siteb = next(
            label for label, entry in key["003-001"].items() if entry["model_id"] == "siteb"
        )
        expected = [
            "images/003-001_{}_001.png".format(label_for_siteb),
            "images/003-001_{}_page.png".format(label_for_siteb),
        ]
        (answer,) = [
            a
            for c in manifest["cases"]
            for a in c["answers"]
            if a["label"] == label_for_siteb
        ]
        self.assertEqual(answer["images"], expected)
        for name in expected:
            self.assertIn(name, names)
        # The other answer has no images.
        other = [
            a for c in manifest["cases"] for a in c["answers"] if a["label"] != label_for_siteb
        ]
        self.assertTrue(all(a["images"] == [] for a in other))

    def test_labels_shuffle_per_case_and_failed_answers_are_excluded(self):
        self.answers.upsert(
            {
                "case_id": "003-002",
                "model_id": "siteb",
                "status": "failed",
                "response_text": "",
                "images": [],
                "error": "boom",
                "case_sha256": "",
            }
        )
        _, key_path = self.build()
        with open(key_path, encoding="utf-8") as f:
            key = json.load(f)["key"]
        self.assertEqual(len(key["003-001"]), 2)  # both ok answers
        self.assertEqual(len(key["003-002"]), 1)  # failed siteb excluded
        self.assertEqual(key["003-002"]["A"]["model_id"], "modela")

    def test_blinding_is_random_per_case(self):
        # With 2 answers per case across many fresh blindings, both orders
        # must appear (probability of missing one is 2^-40).
        seen = set()
        cases = [self.master.cases["003-001"]]
        records = [
            self.answers.get("003-001", "modela"),
            self.answers.get("003-001", "siteb"),
        ]
        for _ in range(40):
            _, key = blind_cases(cases, {"003-001": records})
            seen.add(key["003-001"]["A"]["model_id"])
        self.assertEqual(seen, {"modela", "siteb"})

    def test_missing_image_warns_but_does_not_crash(self):
        os.unlink(self.image)
        warnings = []
        build_package(
            "pkg2", ["003-001"], ["siteb"], self.master, self.answers, warn=warnings.append
        )
        self.assertTrue(any("missing" in w for w in warnings))
        with zipfile.ZipFile(os.path.join("scoring_packages", "pkg2.zip")) as bundle:
            manifest = json.loads(bundle.read("package.json"))
        self.assertEqual(len(manifest["cases"][0]["answers"][0]["images"]), 1)

    def test_blinded_image_name_forms(self):
        self.assertEqual(
            blinded_image_name("003-001", "B", "answer_images/003-001_x_007.png"),
            "images/003-001_B_007.png",
        )
        self.assertEqual(
            blinded_image_name("003-001", "B", "answer_images/003-001_x_page.png"),
            "images/003-001_B_page.png",
        )

    def test_self_identification_scan(self):
        cases = [
            {
                "case_id": "003-001",
                "answers": [
                    {"label": "A", "response_text": "As ChatGPT, I think..."},
                    {"label": "B", "response_text": "The diagnosis is clear."},
                ],
            }
        ]
        warnings = self_identification_warnings(cases)
        self.assertEqual(len(warnings), 1)
        self.assertIn("ChatGPT", warnings[0])
        self.assertIn("answer A", warnings[0].replace("answer A", "answer A"))

    def test_scorer_readme_present(self):
        zip_path, _ = self.build()
        with zipfile.ZipFile(zip_path) as bundle:
            readme = bundle.read("README.txt").decode("utf-8")
        self.assertIn("NOT the same AI", readme)


if __name__ == "__main__":
    unittest.main()
