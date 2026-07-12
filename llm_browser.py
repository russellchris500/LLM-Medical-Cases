#!/usr/bin/env python3
"""Browser automation for the healthcare LLM sites that have no API
(OpenEvidence, UpToDate, Doximity GPT), used by Program 3 (run_llms.py).

Design notes:
- Playwright is imported lazily; every other part of the runner works
  without it. Settings has a one-time setup item that installs it.
- Each site gets its own persistent browser profile under
  browser_profiles/<site>/ so a login (including "remember this device"
  after 2FA) survives between runs. The browser always runs visibly:
  logins, verification codes, and CAPTCHAs are strictly human steps, and
  the manual-assist fallback needs a window the user can see.
- All page selectors are ordered fallback lists, and can be replaced
  without touching code by placing a site_selectors.json file next to the
  program (so a helper can fix a site redesign by emailing one file).
- Any failed step raises BrowserStepError; the runner catches it and, with
  the window left open, offers retry / do-it-by-hand / skip / abandon.
  The do-it-by-hand path needs no working selectors at all, so the
  program stays usable even after a total site redesign.
"""

import base64
import copy
import json
import os
import time
import urllib.parse

try:
    from playwright.sync_api import sync_playwright  # noqa: F401

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    sync_playwright = None
    PLAYWRIGHT_AVAILABLE = False

PROFILES_DIR = "browser_profiles"
SELECTOR_OVERRIDE_FILE = "site_selectors.json"

BROWSER_MODEL_IDS = ["openevidence", "uptodate", "doximity", "chatgptclinicians"]

# Ordered fallback lists: each selector is tried in turn until one matches.
# Written generically (roles, aria labels, broad containers) because the
# sites change without notice; the last answer_container entries are wide
# on purpose so extraction degrades to "grab the whole page" not "crash".
DEFAULT_SELECTORS = {
    "openevidence": {
        "question_box": [
            "textarea",
            "[contenteditable='true']",
            "input[type='text']",
        ],
        "submit_button": [
            "button[type='submit']",
            "[aria-label*='send' i]",
            "[data-testid*='send' i]",
        ],
        "answer_container": [
            "main [class*='answer' i]",
            "main [class*='response' i]",
            "main article",
            "main",
            "body",
        ],
        "login_form": ["input[type='password']"],
        "answer_images": ["img"],
        "username_field": ["input[type='email']", "input[name*='email' i]", "input[type='text']"],
        "password_field": ["input[type='password']"],
        "new_chat": ["[aria-label*='new chat' i]", "[aria-label*='new question' i]", "[data-testid*='new-chat' i]"],
    },
    "uptodate": {
        "question_box": [
            "textarea",
            "input[type='search']",
            "[contenteditable='true']",
            "input[type='text']",
        ],
        "submit_button": [
            "button[type='submit']",
            "[aria-label*='search' i]",
            "[aria-label*='send' i]",
        ],
        "answer_container": [
            "main [class*='answer' i]",
            "main [class*='result' i]",
            "main article",
            "main",
            "body",
        ],
        "login_form": ["input[type='password']"],
        "answer_images": ["img"],
        "username_field": ["input[name*='user' i]", "input[type='email']", "input[type='text']"],
        "password_field": ["input[type='password']"],
        "new_chat": ["[aria-label*='new search' i]", "[aria-label*='new question' i]"],
    },
    "doximity": {
        "question_box": [
            "textarea",
            "[contenteditable='true']",
            "input[type='text']",
        ],
        "submit_button": [
            "button[type='submit']",
            "[aria-label*='send' i]",
        ],
        "answer_container": [
            "main [class*='answer' i]",
            "main [class*='message' i]",
            "main article",
            "main",
            "body",
        ],
        "login_form": ["input[type='password']"],
        "answer_images": ["img"],
        "username_field": ["input[type='email']", "input[name*='email' i]", "input[type='text']"],
        "password_field": ["input[type='password']"],
        "new_chat": ["[aria-label*='new chat' i]", "[aria-label*='new question' i]", "[data-testid*='new-chat' i]"],
    },
    "chatgptclinicians": {
        "question_box": [
            "#prompt-textarea",
            "[contenteditable='true']",
            "textarea",
        ],
        "submit_button": [
            "[data-testid='send-button']",
            "button[aria-label*='send' i]",
            "button[type='submit']",
        ],
        "answer_container": [
            "[data-message-author-role='assistant']",
            "main article",
            "main",
            "body",
        ],
        # Logged-out chatgpt.com still shows a composer, so the login check
        # keys on the login/signup buttons rather than a password field.
        "login_form": [
            "[data-testid='login-button']",
            "button[data-testid='signup-button']",
            "input[type='password']",
        ],
        "answer_images": ["img"],
        "username_field": ["input[type='email']", "input[name*='email' i]"],
        "password_field": ["input[type='password']"],
        "new_chat": [
            "[data-testid='create-new-chat-button']",
            "[aria-label*='new chat' i]",
            "a[href='/']",
        ],
    },
}

SITE_INFO = {
    "openevidence": {
        "display_name": "OpenEvidence",
        "home_url": "https://www.openevidence.com/",
        "login_url": "https://www.openevidence.com/",
    },
    "uptodate": {
        "display_name": "UpToDate",
        "home_url": "https://www.uptodate.com/contents/search",
        "login_url": "https://www.uptodate.com/login",
    },
    "doximity": {
        "display_name": "Doximity GPT",
        "home_url": "https://www.doximity.com/docs-gpt",
        "login_url": "https://www.doximity.com/docs-gpt",
    },
    "chatgptclinicians": {
        "display_name": "ChatGPT for Clinicians",
        "home_url": "https://chatgpt.com/",
        "login_url": "https://chatgpt.com/auth/login",
    },
}


class BrowserStepError(Exception):
    """A single automation step failed; the runner offers recovery options."""

    def __init__(self, step, detail):
        super().__init__(detail)
        self.step = step
        self.detail = detail


def load_selectors(path=SELECTOR_OVERRIDE_FILE):
    """Built-in selectors, optionally patched by site_selectors.json."""
    selectors = copy.deepcopy(DEFAULT_SELECTORS)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                overrides = json.load(f)
        except (OSError, ValueError):
            return selectors
        if isinstance(overrides, dict):
            for site_id, site_overrides in overrides.items():
                if site_id in selectors and isinstance(site_overrides, dict):
                    for key, value in site_overrides.items():
                        if isinstance(value, list) and all(isinstance(s, str) for s in value):
                            selectors[site_id][key] = value
    return selectors


def find_first(page, selector_list):
    """Return the first visible element matching any selector, or None."""
    for selector in selector_list:
        try:
            for element in page.query_selector_all(selector):
                if element.is_visible():
                    return element
        except Exception:
            continue
    return None


EXT_FOR_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
    "image/bmp": ".bmp",
    "image/tiff": ".tif",
}

# In-page fetch used when a direct download is impossible (blob: URLs) -
# runs inside the site's own page, so cookies and blobs both work.
_FETCH_IMAGE_JS = """async e => {
    const src = e.currentSrc || e.src;
    if (!src) return "";
    const resp = await fetch(src);
    const buf = await resp.arrayBuffer();
    const bytes = new Uint8Array(buf);
    let s = "";
    for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
    return btoa(s);
}"""


def sniff_image_extension(data, content_type="", src=""):
    """File extension from the image bytes themselves (most reliable),
    then the Content-Type, then the URL; None if it isn't image data."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:2] == b"\xff\xd8":
        return ".jpg"
    if data[:4] in (b"GIF8",):
        return ".gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if b"<svg" in data[:512].lower():
        return ".svg"
    if data[:2] == b"BM":
        return ".bmp"
    mime = (content_type or "").split(";")[0].strip().lower()
    if mime in EXT_FOR_MIME:
        return EXT_FOR_MIME[mime]
    url_ext = os.path.splitext(urllib.parse.urlparse(src).path)[1].lower()
    if url_ext in EXT_FOR_MIME.values():
        return url_ext
    return None


class BrowserAnswer:
    def __init__(self, text, image_paths, model_reported, manual=False):
        self.text = text
        self.image_paths = image_paths
        self.model_reported = model_reported
        self.manual = manual


class SiteDriver:
    """Drives one site. Subclasses only pin identity; behavior is generic."""

    site_id = None

    def __init__(self, selectors=None):
        info = SITE_INFO[self.site_id]
        self.display_name = info["display_name"]
        self.home_url = info["home_url"]
        self.login_url = info["login_url"]
        all_selectors = selectors or load_selectors()
        self.selectors = all_selectors[self.site_id]

    def model_reported(self):
        return "{} (web interface, accessed {})".format(
            self.display_name, time.strftime("%Y-%m-%d")
        )

    # ---- login ----

    def is_logged_in(self, page):
        """True when the question box is visible and no login form is."""
        box = find_first(page, self.selectors["question_box"])
        login = find_first(page, self.selectors["login_form"])
        return box is not None and login is None

    def autofill_login(self, page, username, password):
        """Best-effort form fill; the human always finishes the login
        (2FA, CAPTCHA, and consent screens are never automated)."""
        try:
            if username:
                field = find_first(page, self.selectors["username_field"])
                if field is not None:
                    field.fill(username)
            if password:
                field = find_first(page, self.selectors["password_field"])
                if field is not None:
                    field.fill(password)
        except Exception:
            pass

    # ---- asking a question ----

    def start_new_question(self, page):
        """Open a fresh conversation so cases never share chat context -
        the sites must have NO MEMORY of earlier cases."""
        try:
            page.goto(self.home_url, wait_until="domcontentloaded")
        except Exception as e:
            raise BrowserStepError(
                "navigation", "could not open {} ({})".format(self.home_url, e)
            )
        # Some sites reopen the previous conversation on their home page;
        # a visible "new chat"-style button is clicked when one exists.
        button = find_first(page, self.selectors.get("new_chat", []))
        if button is not None:
            try:
                button.click()
            except Exception:
                pass

    def baseline_text(self, page):
        """Text already on the page, so old content is never mistaken for
        the new answer."""
        container = find_first(page, self.selectors["answer_container"])
        if container is None:
            return ""
        try:
            return container.inner_text()
        except Exception:
            return ""

    def submit_question(self, page, prompt_text):
        if find_first(page, self.selectors["login_form"]) is not None:
            raise BrowserStepError("logged_out", "the site is showing a login form")
        box = find_first(page, self.selectors["question_box"])
        if box is None:
            raise BrowserStepError(
                "question_box",
                "could not find where to type the question (the site may have "
                "changed its design)",
            )
        try:
            box.click()
            box.fill(prompt_text)
        except Exception as e:
            raise BrowserStepError("question_box", "could not enter the question ({})".format(e))
        button = find_first(page, self.selectors["submit_button"])
        try:
            if button is not None:
                button.click()
            else:
                page.keyboard.press("Enter")
        except Exception as e:
            raise BrowserStepError("submit", "could not send the question ({})".format(e))

    def current_answer_text(self, page, baseline):
        container = find_first(page, self.selectors["answer_container"])
        if container is None:
            return ""
        try:
            text = container.inner_text()
        except Exception:
            return ""
        if text == baseline:
            return ""
        return text

    def wait_for_answer(
        self,
        page,
        baseline,
        stable_seconds=10,
        max_wait_seconds=300,
        poll_seconds=2.0,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        """The answer is done when the page text has appeared and then
        stopped changing for stable_seconds (streaming-safe and site-
        agnostic)."""
        start = clock()
        last_text = None
        stable_since = None
        while True:
            text = self.current_answer_text(page, baseline)
            now = clock()
            if text.strip():
                if text == last_text:
                    if stable_since is not None and now - stable_since >= stable_seconds:
                        return
                else:
                    last_text = text
                    stable_since = now
            if now - start > max_wait_seconds:
                raise BrowserStepError(
                    "waiting",
                    "the answer did not finish appearing within {} seconds".format(
                        max_wait_seconds
                    ),
                )
            sleep(poll_seconds)

    # ---- capturing the answer ----

    def download_image(self, page, element):
        """The actual image file behind an <img>: (bytes, extension), or
        (None, None) so the caller falls back to an element screenshot.

        Three routes, in order: data: URLs are decoded directly; normal
        URLs are downloaded with the browser's own cookies (works behind
        the sites' logins); blob:/other URLs are fetched from inside the
        page itself. Whatever arrives is only accepted if it really is
        image data."""
        try:
            src = element.evaluate("e => e.currentSrc || e.src || ''") or ""
        except Exception:
            src = ""
        if not src:
            return None, None

        if src.startswith("data:"):
            try:
                header, _, payload = src.partition(",")
                if ";base64" in header:
                    data = base64.b64decode(payload)
                else:
                    data = urllib.parse.unquote_to_bytes(payload)
                extension = sniff_image_extension(data, header[5:], src)
                if data and extension:
                    return data, extension
            except Exception:
                pass
            return None, None

        if not src.startswith("blob:"):
            try:
                response = page.request.get(src)
                if response.ok:
                    data = response.body()
                    extension = sniff_image_extension(
                        data, response.headers.get("content-type", ""), src
                    )
                    if data and extension:
                        return data, extension
            except Exception:
                pass

        try:
            encoded = element.evaluate(_FETCH_IMAGE_JS)
            if encoded:
                data = base64.b64decode(encoded)
                extension = sniff_image_extension(data, "", src)
                if data and extension:
                    return data, extension
        except Exception:
            pass
        return None, None

    def extract_answer(self, page, images_dir, basename, baseline="", manual=False):
        container = find_first(page, self.selectors["answer_container"])
        text = ""
        if container is not None:
            try:
                text = container.inner_text()
            except Exception:
                text = ""
        if not text.strip():
            try:
                body = page.query_selector("body")
                text = body.inner_text() if body is not None else ""
            except Exception:
                text = ""
        if baseline and text == baseline:
            text = ""

        os.makedirs(images_dir, exist_ok=True)
        image_paths = []
        scope = container if container is not None else page
        counter = 0
        for selector in self.selectors["answer_images"]:
            try:
                elements = scope.query_selector_all(selector)
            except Exception:
                continue
            for element in elements:
                try:
                    box = element.bounding_box()
                    # Skip icons/avatars; keep figures the physician would read.
                    if not box or box["width"] < 100 or box["height"] < 100:
                        continue
                    counter += 1
                    # The REAL image file first (downloaded through the
                    # site's own logged-in session); a screenshot of the
                    # element is only the last resort.
                    data, extension = self.download_image(page, element)
                    if data:
                        path = os.path.join(
                            images_dir, "{}_{:03d}{}".format(basename, counter, extension)
                        )
                        with open(path, "wb") as f:
                            f.write(data)
                    else:
                        path = os.path.join(
                            images_dir, "{}_{:03d}.png".format(basename, counter)
                        )
                        element.screenshot(path=path)
                    image_paths.append(path)
                except Exception:
                    continue
            if image_paths:
                break  # first matching selector family is enough

        # Full-page screenshot: the audit trail even if text extraction
        # grabbed the wrong container.
        page_path = os.path.join(images_dir, "{}_page.png".format(basename))
        try:
            page.screenshot(path=page_path, full_page=True)
            image_paths.append(page_path)
        except Exception:
            pass

        if not text.strip():
            raise BrowserStepError(
                "extraction",
                "no answer text could be read from the page (a full-page "
                "screenshot was saved)",
            )
        return BrowserAnswer(text, image_paths, self.model_reported(), manual=manual)


class OpenEvidenceDriver(SiteDriver):
    site_id = "openevidence"


class UpToDateDriver(SiteDriver):
    site_id = "uptodate"


class DoximityDriver(SiteDriver):
    site_id = "doximity"


class ChatGPTCliniciansDriver(SiteDriver):
    site_id = "chatgptclinicians"

    def is_logged_in(self, page):
        """chatgpt.com shows a composer even when logged out, so being
        logged in means the login/signup buttons are gone AND the composer
        is there. The clinician workspace also requires the right account,
        which only the user can confirm - the login flow asks them."""
        return super().is_logged_in(page)


DRIVER_CLASSES = {
    "openevidence": OpenEvidenceDriver,
    "uptodate": UpToDateDriver,
    "doximity": DoximityDriver,
    "chatgptclinicians": ChatGPTCliniciansDriver,
}


def make_driver(site_id, selectors=None):
    return DRIVER_CLASSES[site_id](selectors)


def open_site_context(playwright, site_id, headless=False):
    """A visible browser with a persistent per-site profile, so logins
    (including remembered 2FA devices) survive between runs.

    Google refuses OAuth sign-ins ("This browser or app may not be
    secure") in browsers that advertise automation, so the automation
    banner/flag is switched off and a real installed Chrome (or Edge,
    which every Windows machine has) is preferred over the bundled
    test browser."""
    user_data_dir = os.path.join(PROFILES_DIR, site_id)
    os.makedirs(user_data_dir, exist_ok=True)
    kwargs = {
        "headless": headless,
        "viewport": {"width": 1280, "height": 900},
        "permissions": ["clipboard-read", "clipboard-write"],
        # Without these, Chromium announces itself as automated
        # (navigator.webdriver + the "controlled by automated test
        # software" bar), which trips Google's unsafe-browser block.
        "ignore_default_args": ["--enable-automation"],
        "args": [
            "--disable-blink-features=AutomationControlled",
            "--no-first-run",
            "--no-default-browser-check",
        ],
    }
    context = None
    for channel in ("chrome", "msedge"):
        try:
            context = playwright.chromium.launch_persistent_context(
                user_data_dir, channel=channel, **kwargs
            )
            break
        except Exception:
            continue
    if context is None:
        context = playwright.chromium.launch_persistent_context(user_data_dir, **kwargs)
    try:
        # Belt and suspenders for pages that probe navigator.webdriver.
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
    except Exception:
        pass
    return context


def copy_to_clipboard(page, text):
    """Put text on the clipboard for the manual-assist path; False if the
    browser refused (caller then prints the text for manual copying)."""
    try:
        page.evaluate("t => navigator.clipboard.writeText(t)", text)
        return True
    except Exception:
        return False
