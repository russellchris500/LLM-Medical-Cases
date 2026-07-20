"""HTTP client the local Runner uses to talk to the Study Hub.

Standard library only (urllib), like the rest of the Runner: physicians
install nothing extra. Authentication is a per-device runner token from
the hub's "Runner tokens" page; API keys and site logins stay local.
"""

import json
import mimetypes
import os
import secrets
import urllib.error
import urllib.request


class HubError(Exception):
    """The hub could not be reached or refused the request."""


def encode_multipart(fields, files):
    """(body_bytes, content_type) for a multipart/form-data request.

    fields: {name: str}; files: list of (name, filename, bytes)."""
    boundary = "----studyhub" + secrets.token_hex(16)
    parts = []
    for name, value in fields.items():
        parts.append(
            "--{}\r\nContent-Disposition: form-data; name=\"{}\"\r\n\r\n{}\r\n".format(
                boundary, name, value
            ).encode("utf-8")
        )
    for name, filename, payload in files:
        content_type = (
            mimetypes.guess_type(filename)[0] or "application/octet-stream"
        )
        parts.append(
            "--{}\r\nContent-Disposition: form-data; name=\"{}\"; "
            "filename=\"{}\"\r\nContent-Type: {}\r\n\r\n".format(
                boundary, name, filename, content_type
            ).encode("utf-8")
        )
        parts.append(payload)
        parts.append(b"\r\n")
    parts.append("--{}--\r\n".format(boundary).encode("utf-8"))
    return b"".join(parts), "multipart/form-data; boundary=" + boundary


class HubClient:
    def __init__(self, base_url, token, urlopen=None, timeout=60):
        self.base_url = (base_url or "").rstrip("/")
        self.token = (token or "").strip()
        self._urlopen = urlopen or urllib.request.urlopen
        self.timeout = timeout

    def configured(self):
        return bool(self.base_url and self.token)

    def _request(self, method, path, body=None, content_type=None):
        if not self.configured():
            raise HubError(
                "The hub is not set up yet - enter the hub address and a "
                "runner token in Settings."
            )
        request = urllib.request.Request(
            self.base_url + path, data=body, method=method,
            headers={"Authorization": "Bearer " + self.token},
        )
        if content_type:
            request.add_header("Content-Type", content_type)
        try:
            with self._urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = ""
            try:
                detail = json.loads(error.read().decode("utf-8")).get("error", "")
            except Exception:
                pass
            if error.code == 401:
                raise HubError(
                    "The hub rejected the runner token - create a fresh one "
                    "on the hub's Runner tokens page. {}".format(detail).strip()
                )
            raise HubError("The hub answered HTTP {}. {}".format(
                error.code, detail
            ).strip())
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise HubError("Could not reach the hub at {} ({}).".format(
                self.base_url, getattr(error, "reason", error)
            ))
        except ValueError:
            raise HubError("The hub sent back something that was not JSON.")

    def jobs(self):
        return self._request("GET", "/api/runner/jobs")["jobs"]

    def upload_answer(self, job_id, record, image_paths=(), html_path=None):
        fields = {
            "run_job_id": str(job_id),
            "case_id": record["case_id"],
            "llm_id": record.get("llm_id", ""),
            "model_name": record.get("model_name", ""),
            "variant_id": record.get("model_id", ""),
            "model_display_name": record.get("model_display_name", ""),
            "response_text": record.get("response_text", ""),
            "thinking_setting": record.get("thinking_setting", ""),
            "model_reported": record.get("model_reported", ""),
            "deep_thinking": "1" if record.get("deep_thinking", True) else "0",
            "status": record.get("status", "ok"),
            "error": record.get("error") or "",
            "case_text_sha256": record.get("case_text_sha256", ""),
            "rubric_version": str(record.get("rubric_version", 1)),
        }
        files = []
        for path in image_paths:
            try:
                with open(path, "rb") as f:
                    files.append(("images", os.path.basename(path), f.read()))
            except OSError:
                continue
        if html_path:
            try:
                with open(html_path, "rb") as f:
                    files.append(
                        ("answer_html", os.path.basename(html_path), f.read())
                    )
            except OSError:
                pass
        body, content_type = encode_multipart(fields, files)
        return self._request("POST", "/api/runner/answers", body, content_type)

    def set_job_status(self, job_id, status, note=""):
        body = json.dumps({"status": status, "note": note}).encode("utf-8")
        return self._request(
            "POST", "/api/runner/jobs/{}/status".format(job_id),
            body, "application/json",
        )
