"""Tests for llm_browser.py using fake page objects (no Playwright needed).
Run with:  python3 -m unittest test_llm_browser.py"""

import json
import os
import tempfile
import unittest

from llm_browser import (
    BrowserStepError,
    DEFAULT_SELECTORS,
    find_first,
    load_selectors,
    make_driver,
    sniff_image_extension,
)


class FakeElement:
    def __init__(self, text="", visible=True, box=(200, 200), attrs=None):
        self.text = text
        self.visible = visible
        self.box = box
        self.attrs = attrs or {}
        self.filled = None
        self.clicked = False
        self.screenshot_paths = []

    def is_visible(self):
        return self.visible

    def inner_text(self):
        return self.text

    def fill(self, value):
        self.filled = value

    def click(self):
        self.clicked = True

    def bounding_box(self):
        if self.box is None:
            return None
        return {"x": 0, "y": 0, "width": self.box[0], "height": self.box[1]}

    def get_attribute(self, name):
        return self.attrs.get(name)

    def evaluate(self, script):
        if "currentSrc" in script and "fetch" not in script:
            return self.attrs.get("src", "")
        if "fetch" in script:
            payload = self.attrs.get("fetch_bytes")
            if payload is None:
                raise RuntimeError("no fetch payload")
            import base64
            return base64.b64encode(payload).decode()
        return None

    def screenshot(self, path=None):
        self.screenshot_paths.append(path)
        with open(path, "wb") as f:
            f.write(b"\x89PNG fake")

    def query_selector_all(self, selector):
        return self.attrs.get("children", {}).get(selector, [])


class FakeResponse:
    def __init__(self, data, content_type="image/png", ok=True):
        self.data = data
        self.ok = ok
        self.headers = {"content-type": content_type}

    def body(self):
        return self.data


class FakeRequest:
    def __init__(self, responses):
        self.responses = responses  # src -> FakeResponse or Exception
        self.asked = []

    def get(self, src):
        self.asked.append(src)
        result = self.responses[src]
        if isinstance(result, Exception):
            raise result
        return result


class FakeKeyboard:
    def __init__(self):
        self.pressed = []

    def press(self, key):
        self.pressed.append(key)


class FakePage:
    """Selector map: selector string -> list of FakeElements."""

    def __init__(self, elements=None):
        self.elements = elements or {}
        self.keyboard = FakeKeyboard()
        self.goto_urls = []
        self.screenshots = []

    def goto(self, url, wait_until=None, timeout=None):
        self.goto_urls.append(url)

    def query_selector(self, selector):
        matches = self.elements.get(selector, [])
        return matches[0] if matches else None

    def query_selector_all(self, selector):
        return self.elements.get(selector, [])

    def screenshot(self, path=None, full_page=False):
        self.screenshots.append(path)
        with open(path, "wb") as f:
            f.write(b"\x89PNG fake page")


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FindFirstTests(unittest.TestCase):
    def test_falls_back_and_skips_invisible(self):
        hidden = FakeElement(visible=False)
        shown = FakeElement()
        page = FakePage({"textarea": [hidden], "[contenteditable='true']": [shown]})
        self.assertIs(find_first(page, ["textarea", "[contenteditable='true']"]), shown)

    def test_none_when_nothing_matches(self):
        self.assertIsNone(find_first(FakePage(), ["textarea"]))


class SelectorOverrideTests(unittest.TestCase):
    def test_override_file_replaces_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "site_selectors.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"openevidence": {"question_box": ["#ask"]}}, f)
            selectors = load_selectors(path)
        self.assertEqual(selectors["openevidence"]["question_box"], ["#ask"])
        # Untouched keys keep their defaults.
        self.assertEqual(
            selectors["openevidence"]["login_form"],
            DEFAULT_SELECTORS["openevidence"]["login_form"],
        )
        self.assertEqual(selectors["uptodate"], DEFAULT_SELECTORS["uptodate"])

    def test_bad_override_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "site_selectors.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{not json")
            selectors = load_selectors(path)
        self.assertEqual(selectors, DEFAULT_SELECTORS)


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.driver = make_driver("openevidence")

    def test_is_logged_in(self):
        page = FakePage({"textarea": [FakeElement()]})
        self.assertTrue(self.driver.is_logged_in(page))
        page = FakePage(
            {"textarea": [FakeElement()], "input[type='password']": [FakeElement()]}
        )
        self.assertFalse(self.driver.is_logged_in(page))
        self.assertFalse(self.driver.is_logged_in(FakePage()))

    def test_start_new_question_clicks_new_chat_button(self):
        # No memory between cases: after navigating home, a visible
        # "new chat" button is clicked so the site can't resume the
        # previous conversation.
        button = FakeElement()
        page = FakePage({"[aria-label*='new chat' i]": [button]})
        self.driver.start_new_question(page)
        self.assertEqual(page.goto_urls, [self.driver.home_url])
        self.assertTrue(button.clicked)
        # And it is simply skipped when the site has no such button.
        self.driver.start_new_question(FakePage())

    def test_submit_question_fills_and_clicks(self):
        box = FakeElement()
        button = FakeElement()
        page = FakePage({"textarea": [box], "button[type='submit']": [button]})
        self.driver.submit_question(page, "The case text")
        self.assertEqual(box.filled, "The case text")
        self.assertTrue(button.clicked)

    def test_submit_falls_back_to_enter_key(self):
        page = FakePage({"textarea": [FakeElement()]})
        self.driver.submit_question(page, "Q")
        self.assertEqual(page.keyboard.pressed, ["Enter"])

    def test_submit_detects_logged_out(self):
        page = FakePage(
            {"textarea": [FakeElement()], "input[type='password']": [FakeElement()]}
        )
        with self.assertRaises(BrowserStepError) as ctx:
            self.driver.submit_question(page, "Q")
        self.assertEqual(ctx.exception.step, "logged_out")

    def test_submit_without_question_box_raises(self):
        with self.assertRaises(BrowserStepError) as ctx:
            self.driver.submit_question(FakePage(), "Q")
        self.assertEqual(ctx.exception.step, "question_box")

    def test_wait_for_answer_stability(self):
        container = FakeElement(text="old content")
        page = FakePage({"main": [container]})
        baseline = self.driver.baseline_text(page)
        clock = FakeClock()
        # Streams: grows twice, then holds still long enough.
        script = iter(
            ["old content", "old content", "Ans", "Answer gro", "Answer grown.",
             "Answer grown.", "Answer grown.", "Answer grown.", "Answer grown.",
             "Answer grown.", "Answer grown."]
        )

        def sleep(seconds):
            clock.sleep(seconds)
            try:
                container.text = next(script)
            except StopIteration:
                pass

        self.driver.wait_for_answer(
            page, baseline, stable_seconds=6, max_wait_seconds=60,
            poll_seconds=2, sleep=sleep, clock=clock,
        )
        self.assertEqual(container.text, "Answer grown.")

    def test_wait_for_answer_times_out(self):
        container = FakeElement(text="same forever")
        page = FakePage({"main": [container]})
        clock = FakeClock()
        with self.assertRaises(BrowserStepError) as ctx:
            self.driver.wait_for_answer(
                page, baseline="same forever", stable_seconds=10,
                max_wait_seconds=30, poll_seconds=2, sleep=clock.sleep, clock=clock,
            )
        self.assertEqual(ctx.exception.step, "waiting")

    def test_extract_answer_text_images_and_page_shot(self):
        with tempfile.TemporaryDirectory() as tmp:
            big = FakeElement(box=(400, 300))
            small = FakeElement(box=(32, 32))
            container = FakeElement(
                text="The answer.", attrs={"children": {"img": [big, small]}}
            )
            page = FakePage({"main": [container]})
            answer = self.driver.extract_answer(page, tmp, "003-001_openevidence")
            self.assertEqual(answer.text, "The answer.")
            names = [os.path.basename(p) for p in answer.image_paths]
            self.assertEqual(
                names, ["003-001_openevidence_001.png", "003-001_openevidence_page.png"]
            )
            for path in answer.image_paths:
                self.assertTrue(os.path.exists(path))
            self.assertIn("OpenEvidence (web interface, accessed", answer.model_reported)

    def test_extract_answer_falls_back_to_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            body = FakeElement(text="Whole page text.")
            page = FakePage({"body": [body]})
            answer = self.driver.extract_answer(page, tmp, "x")
            self.assertEqual(answer.text, "Whole page text.")

    def test_extract_answer_no_text_raises_but_saves_screenshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = FakePage()
            with self.assertRaises(BrowserStepError) as ctx:
                self.driver.extract_answer(page, tmp, "x")
            self.assertEqual(ctx.exception.step, "extraction")
            self.assertTrue(os.path.exists(os.path.join(tmp, "x_page.png")))

    def test_all_sites_have_drivers_and_selectors(self):
        from llm_browser import BROWSER_MODEL_IDS
        for site_id in ("chatgptclinicians", "amboss", "clinicalkeyai", "dynamed", "glasshealth"):
            self.assertIn(site_id, BROWSER_MODEL_IDS)
        for site_id in BROWSER_MODEL_IDS:
            driver = make_driver(site_id)
            self.assertTrue(driver.home_url.startswith("https://"))
            for key in ("question_box", "submit_button", "answer_container", "login_form"):
                self.assertTrue(driver.selectors[key], (site_id, key))


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fakepngdata" * 20
JPG_BYTES = b"\xff\xd8\xff\xe0" + b"fakejpgdata" * 20


class ImageDownloadTests(unittest.TestCase):
    def setUp(self):
        self.driver = make_driver("openevidence")

    def test_sniffing(self):
        self.assertEqual(sniff_image_extension(PNG_BYTES), ".png")
        self.assertEqual(sniff_image_extension(JPG_BYTES), ".jpg")
        self.assertEqual(sniff_image_extension(b"junk", "image/webp"), ".webp")
        self.assertEqual(
            sniff_image_extension(b"junk", "", "https://x/fig.gif?v=2"), ".gif"
        )
        self.assertIsNone(sniff_image_extension(b"<html>not an image"))

    def test_extract_downloads_real_file_via_browser_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = FakeElement(box=(400, 300),
                              attrs={"src": "https://site/fig.jpg"})
            container = FakeElement(text="Answer.",
                                    attrs={"children": {"img": [img]}})
            page = FakePage({"main": [container]})
            page.request = FakeRequest(
                {"https://site/fig.jpg": FakeResponse(JPG_BYTES, "image/jpeg")}
            )
            answer = self.driver.extract_answer(page, tmp, "003-001_x")
            names = sorted(os.path.basename(p) for p in answer.image_paths)
            self.assertEqual(names, ["003-001_x_001.jpg", "003-001_x_page.png"])
            with open(os.path.join(tmp, "003-001_x_001.jpg"), "rb") as f:
                self.assertEqual(f.read(), JPG_BYTES)   # the ACTUAL file
            self.assertEqual(img.screenshot_paths, [])  # no screenshot needed

    def test_extract_decodes_data_uri(self):
        import base64
        with tempfile.TemporaryDirectory() as tmp:
            src = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()
            img = FakeElement(box=(200, 150), attrs={"src": src})
            container = FakeElement(text="Answer.",
                                    attrs={"children": {"img": [img]}})
            page = FakePage({"main": [container]})
            answer = self.driver.extract_answer(page, tmp, "c_x")
            with open(os.path.join(tmp, "c_x_001.png"), "rb") as f:
                self.assertEqual(f.read(), PNG_BYTES)

    def test_extract_uses_in_page_fetch_for_blob_urls(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = FakeElement(box=(300, 200),
                              attrs={"src": "blob:https://site/abc",
                                     "fetch_bytes": PNG_BYTES})
            container = FakeElement(text="Answer.",
                                    attrs={"children": {"img": [img]}})
            page = FakePage({"main": [container]})
            answer = self.driver.extract_answer(page, tmp, "c_y")
            with open(os.path.join(tmp, "c_y_001.png"), "rb") as f:
                self.assertEqual(f.read(), PNG_BYTES)

    def test_extract_falls_back_to_screenshot_when_download_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = FakeElement(box=(400, 300),
                              attrs={"src": "https://site/fig.png"})
            container = FakeElement(text="Answer.",
                                    attrs={"children": {"img": [img]}})
            page = FakePage({"main": [container]})
            page.request = FakeRequest(
                {"https://site/fig.png": RuntimeError("403 forbidden")}
            )
            answer = self.driver.extract_answer(page, tmp, "c_z")
            self.assertTrue(any(p.endswith("c_z_001.png") for p in answer.image_paths))
            self.assertEqual(len(img.screenshot_paths), 1)  # screenshot fallback

    def test_non_image_response_rejected_then_screenshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = FakeElement(box=(400, 300),
                              attrs={"src": "https://site/paywall"})
            container = FakeElement(text="Answer.",
                                    attrs={"children": {"img": [img]}})
            page = FakePage({"main": [container]})
            page.request = FakeRequest(
                {"https://site/paywall": FakeResponse(b"<html>login</html>", "text/html")}
            )
            answer = self.driver.extract_answer(page, tmp, "c_w")
            self.assertEqual(len(img.screenshot_paths), 1)



if __name__ == "__main__":
    unittest.main()
