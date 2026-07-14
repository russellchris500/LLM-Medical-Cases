#!/usr/bin/env python3
"""Browser automation for the healthcare LLM sites that have no API
(OpenEvidence, UpToDate, Doximity GPT, ChatGPT for Clinicians, AMBOSS,
ClinicalKey AI, DynaMed, Glass Health, the GPT-OSS playground), used by
Program 3 (run_llms.py).

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
import difflib
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

BROWSER_MODEL_IDS = [
    "openevidence",
    "uptodate",
    "doximity",
    "chatgptclinicians",
    "amboss",
    "clinicalkeyai",
    "dynamed",
    "glasshealth",
    "gptoss",
]

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
    "amboss": {
        "question_box": [
            "textarea",
            "[contenteditable='true']",
            "input[type='search']",
            "input[type='text']",
        ],
        "submit_button": [
            "button[type='submit']",
            "[aria-label*='send' i]",
            "[aria-label*='search' i]",
            "[data-testid*='send' i]",
        ],
        "answer_container": [
            "main [class*='answer' i]",
            "main [class*='response' i]",
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
    "clinicalkeyai": {
        "question_box": [
            "textarea",
            "[contenteditable='true']",
            "input[type='search']",
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
            "main [class*='message' i]",
            "main article",
            "main",
            "body",
        ],
        # ClinicalKey AI signs in by email link / institutional SSO, so a
        # password box may never appear; the missing question box is what
        # marks the logged-out state.
        "login_form": ["input[type='password']"],
        "answer_images": ["img"],
        "username_field": ["input[type='email']", "input[name*='email' i]", "input[type='text']"],
        "password_field": ["input[type='password']"],
        "new_chat": ["[aria-label*='new chat' i]", "[aria-label*='new search' i]", "[data-testid*='new-chat' i]"],
    },
    "dynamed": {
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
            "main [class*='response' i]",
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
    "glasshealth": {
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
    "gptoss": {
        "question_box": [
            "textarea",
            "[contenteditable='true']",
            "[role='textbox']",
            "input[type='text']",
            "input:not([type])",
        ],
        "submit_button": [
            "button[type='submit']",
            "[aria-label*='send' i]",
            "[data-testid*='send' i]",
        ],
        "answer_container": [
            "main [class*='answer' i]",
            "main [class*='response' i]",
            "main [class*='message' i]",
            "main article",
            "main",
            "body",
        ],
        # The playground has no accounts at all, so no login form ever
        # appears and the site counts as logged in whenever the question
        # box is on screen.
        "login_form": ["input[type='password']"],
        "answer_images": ["img"],
        "username_field": ["input[type='email']"],
        "password_field": ["input[type='password']"],
        "new_chat": ["[aria-label*='new chat' i]", "[data-testid*='new-chat' i]", "a[href='/']"],
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
    "amboss": {
        "display_name": "AMBOSS",
        "home_url": "https://next.amboss.com/us",
        "login_url": "https://next.amboss.com/us/login",
    },
    "clinicalkeyai": {
        "display_name": "ClinicalKey AI",
        "home_url": "https://ai.clinicalkey.com/",
        "login_url": "https://ai.clinicalkey.com/",
    },
    "dynamed": {
        "display_name": "DynaMed",
        "home_url": "https://www.dynamed.com/",
        "login_url": "https://www.dynamed.com/login",
    },
    "glasshealth": {
        "display_name": "Glass Health",
        "home_url": "https://glass.health/",
        "login_url": "https://glass.health/login",
    },
    "gptoss": {
        "display_name": "GPT-OSS Playground",
        "home_url": "https://gpt-oss.com/",
        "login_url": "https://gpt-oss.com/",
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


def _search_surfaces(page):
    """The page plus every iframe in it - some sites build their whole
    chat UI inside a frame, where a plain page query finds nothing."""
    frames = getattr(page, "frames", None)
    if frames:
        try:
            surfaces = list(frames)  # includes the main frame
            if surfaces:
                return surfaces
        except Exception:
            pass
    return [page]


def find_first_located(page, selector_list):
    """Return (element, surface) for the first visible element matching
    any selector, searching the page and any iframes; (None, None) if
    nothing matches. The surface is the page or frame the element lives
    in, so later lookups can stay inside the same document."""
    surfaces = _search_surfaces(page)
    for selector in selector_list:
        for surface in surfaces:
            try:
                for element in surface.query_selector_all(selector):
                    if element.is_visible():
                        return element, surface
            except Exception:
                continue
    return None, None


def find_first(page, selector_list):
    """Return the first visible element matching any selector, searching
    the page and any iframes, or None."""
    element, _ = find_first_located(page, selector_list)
    return element


_DESCRIBE_JS = """() => {
    const rows = [];
    Array.from(document.querySelectorAll(
        "textarea, input, [contenteditable], [role='textbox'], button, form"
    )).slice(0, 80).forEach(e => rows.push([
        e.tagName.toLowerCase(),
        e.getAttribute('type') || '',
        e.id || '',
        e.getAttribute('placeholder') || '',
        e.getAttribute('aria-label') || '',
        e.getAttribute('data-testid') || '',
        (e.offsetWidth || e.offsetHeight) ? 'visible' : 'hidden',
    ].join(' | ')));
    rows.push('--- content areas (tag | class | text length) ---');
    Array.from(document.querySelectorAll(
        "main, article, [class*='answer' i], [class*='response' i], " +
        "[class*='message' i], [class*='chat' i]"
    )).slice(0, 80).forEach(e => rows.push([
        e.tagName.toLowerCase(),
        String(e.className || '').slice(0, 70),
        String((e.innerText || '').length) + ' chars',
    ].join(' | ')));
    return rows;
}"""


# Every piece of text on the page, including inside shadow DOM widgets
# that innerText cannot see into.
_DEEP_TEXT_JS = """() => {
    const parts = [];
    const scan = (scope) => {
        for (const el of scope.querySelectorAll('*')) {
            if (el.shadowRoot) {
                parts.push(el.shadowRoot.textContent || '');
                scan(el.shadowRoot);
            }
        }
    };
    parts.push(document.body ? (document.body.innerText || '') : '');
    if (document.body) scan(document.body);
    return parts.join('\\n');
}"""


def added_text(before, after):
    """The lines of `after` that were not in `before` - the streamed-in
    answer, on pages where no container selector fits."""
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    matcher = difflib.SequenceMatcher(None, before_lines, after_lines, autojunk=False)
    added = []
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in ("insert", "replace"):
            added.extend(after_lines[j1:j2])
    return "\n".join(line for line in added if line.strip())


def describe_page(page, path):
    """Write a plain-text report of what the page is showing (each frame
    and its input-like elements). When a site check misbehaves, this file
    is everything a helper needs to fix the selectors remotely."""
    lines = []
    for index, surface in enumerate(_search_surfaces(page)):
        try:
            url = surface.url
        except Exception:
            url = "(unknown)"
        lines.append("--- frame {}: {}".format(index, url))
        lines.append("    tag | type | id | placeholder | aria-label | data-testid | visible")
        try:
            for row in surface.evaluate(_DESCRIBE_JS):
                lines.append("    " + row)
        except Exception as e:
            lines.append("    (could not inspect this frame: {})".format(e))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


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
    def __init__(self, text, image_paths, model_reported, manual=False, html_path=None):
        self.text = text
        self.image_paths = image_paths
        self.model_reported = model_reported
        self.manual = manual
        # A saved copy of the answer's real HTML, for formatted reading.
        self.html_path = html_path


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
        # Modern sites draw their page with JavaScript well after the
        # navigation itself "finishes"; how long to keep looking for the
        # page content before concluding it isn't there.
        self.ready_timeout_s = 25
        self.ready_poll_s = 0.5
        # Snapshot of ALL page text taken just before a question is sent,
        # for the diff-based capture fallback (see current_answer_text).
        self._deep_baseline = ""
        # The question last submitted, so its echo in the chat is never
        # mistaken for the beginning of an answer.
        self._last_prompt = ""

    def wait_until_ready(self, page, sleep=time.sleep, clock=time.monotonic):
        """Wait until the page has actually drawn a question box or a login
        form. On sites rendered entirely with JavaScript (gpt-oss.com,
        chatgpt.com), the page is still blank at the moment navigation
        completes - checking right away sees nothing and wrongly concludes
        the user is not logged in."""
        start = clock()
        while True:
            if find_first(page, self.selectors["question_box"]) is not None:
                return True
            if find_first(page, self.selectors["login_form"]) is not None:
                return True
            if clock() - start >= self.ready_timeout_s:
                return False
            sleep(self.ready_poll_s)

    def model_reported(self):
        return "{} (web interface, accessed {})".format(
            self.display_name, time.strftime("%Y-%m-%d")
        )

    # ---- login ----

    def is_logged_in(self, page, sleep=time.sleep, clock=time.monotonic):
        """True when the question box is visible and no login form is,
        after giving the page time to render itself."""
        self.wait_until_ready(page, sleep=sleep, clock=clock)
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
        self.wait_until_ready(page)
        # Some sites reopen the previous conversation on their home page;
        # a visible "new chat"-style button is clicked when one exists.
        button = find_first(page, self.selectors.get("new_chat", []))
        if button is not None:
            try:
                button.click()
            except Exception:
                pass

    def chat_surface(self, page):
        """The page or iframe that actually holds the chat. Generic
        answer selectors like 'main' or 'body' also match the outer page
        shell on sites that embed their chat in a frame, so the answer
        must always be read from the same document as the question box."""
        _, surface = find_first_located(page, self.selectors["question_box"])
        return surface if surface is not None else page

    def deep_text(self, page):
        """All text on every surface, including shadow DOM content."""
        parts = []
        for surface in _search_surfaces(page):
            try:
                parts.append(surface.evaluate(_DEEP_TEXT_JS) or "")
            except Exception:
                continue
        return "\n".join(parts)

    def baseline_text(self, page):
        """Text already on the page, so old content is never mistaken for
        the new answer. Also snapshots the whole page's text for the
        diff-based capture fallback."""
        self._deep_baseline = self.deep_text(page)
        surface = self.chat_surface(page)
        container = find_first(surface, self.selectors["answer_container"])
        if container is None:
            return ""
        try:
            return container.inner_text()
        except Exception:
            return ""

    def submit_question(self, page, prompt_text, sleep=time.sleep, clock=time.monotonic):
        self._last_prompt = prompt_text
        self.wait_until_ready(page, sleep=sleep, clock=clock)
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
        """The combined 'what has appeared so far' signal.

        Both the container text AND all text added anywhere on the page
        are included: on some sites the container only ever shows the
        echoed question while the real answer streams into an element no
        selector matches - judged on the container alone, the page looks
        'stable' seconds after submitting and the wait ends while the
        model is still writing."""
        surface = self.chat_surface(page)
        container = find_first(surface, self.selectors["answer_container"])
        text = ""
        if container is not None:
            try:
                text = container.inner_text()
            except Exception:
                text = ""
        if text == baseline:
            text = ""
        diff_text = added_text(self._deep_baseline, self.deep_text(page))
        if diff_text:
            text = (text + "\n" + diff_text) if text else diff_text
        # The echoed question appears within a second of submitting; if it
        # counted as "the answer has started", a model that thinks quietly
        # for a while would look finished before writing a single word.
        if text and self._last_prompt:
            prompt_lines = set(self._last_prompt.splitlines())
            text = "\n".join(
                line for line in text.splitlines()
                if line.strip() and line not in prompt_lines
            )
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

    def extract_answer(self, page, images_dir, basename, baseline="", manual=False,
                       prompt_text=""):
        surface = self.chat_surface(page)
        container = find_first(surface, self.selectors["answer_container"])
        text = ""
        if container is not None:
            try:
                text = container.inner_text()
            except Exception:
                text = ""
        if baseline and text == baseline:
            text = ""
        # Whatever text was added anywhere on the page since just before
        # the question was sent (the echoed question itself is dropped).
        # If the page gained answer text the container never showed, the
        # container missed the answer - trust the page diff instead.
        diff_text = added_text(self._deep_baseline, self.deep_text(page))
        if prompt_text and diff_text:
            prompt_lines = set(prompt_text.splitlines())
            diff_text = "\n".join(
                line for line in diff_text.splitlines()
                if line not in prompt_lines
            )
        container_missed = False
        if diff_text.strip():
            container_lines = set(text.splitlines())
            missed = [
                line for line in diff_text.splitlines()
                if line.strip() and line not in container_lines
            ]
            # Chat pages always gain a little chrome outside the answer
            # (a new sidebar title, a "said:" label); that must not throw
            # away a good container capture. The diff only wins when the
            # container missed MORE than it holds - e.g. it shows just the
            # echoed question while the whole answer streamed elsewhere.
            prompt_lines = set(prompt_text.splitlines()) if prompt_text else set()
            container_content = sum(
                len(line) for line in text.splitlines()
                if line.strip() and line not in prompt_lines
            )
            if missed and sum(len(line) for line in missed) > container_content:
                text = diff_text
                container_missed = True
        if not text.strip():
            try:
                body = surface.query_selector("body")
                text = body.inner_text() if body is not None else ""
            except Exception:
                text = ""
        if baseline and text == baseline:
            text = ""

        os.makedirs(images_dir, exist_ok=True)
        image_paths = []
        scope = container if container is not None else surface
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

        # Save the answer's real HTML too, so it can be READ with its
        # formatting (headings, bold, tables) in a web browser. When the
        # container missed the answer, the whole document is kept instead.
        html_path = None
        html_source = container
        if html_source is None or container_missed:
            try:
                html_source = surface.query_selector("body")
            except Exception:
                html_source = None
        if html_source is not None:
            try:
                html = html_source.evaluate("e => e.outerHTML") or ""
            except Exception:
                html = ""
            if html.strip():
                html_path = os.path.join(images_dir, "{}_answer.html".format(basename))
                document = (
                    "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                    "<title>{} answer</title><base href=\"{}\"></head>"
                    "<body>{}</body></html>"
                ).format(self.display_name, self.home_url, html)
                try:
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(document)
                except OSError:
                    html_path = None

        if not text.strip():
            raise BrowserStepError(
                "extraction",
                "no answer text could be read from the page (a full-page "
                "screenshot was saved)",
            )
        return BrowserAnswer(
            text, image_paths, self.model_reported(), manual=manual, html_path=html_path
        )


class OpenEvidenceDriver(SiteDriver):
    site_id = "openevidence"


class UpToDateDriver(SiteDriver):
    site_id = "uptodate"


class DoximityDriver(SiteDriver):
    site_id = "doximity"


class ChatGPTCliniciansDriver(SiteDriver):
    site_id = "chatgptclinicians"

    def is_logged_in(self, page, **kwargs):
        """chatgpt.com shows a composer even when logged out, so being
        logged in means the login/signup buttons are gone AND the composer
        is there. The clinician workspace also requires the right account,
        which only the user can confirm - the login flow asks them."""
        return super().is_logged_in(page, **kwargs)


class AmbossDriver(SiteDriver):
    site_id = "amboss"


class ClinicalKeyAIDriver(SiteDriver):
    site_id = "clinicalkeyai"


class DynaMedDriver(SiteDriver):
    site_id = "dynamed"


class GlassHealthDriver(SiteDriver):
    site_id = "glasshealth"


class GptOssDriver(SiteDriver):
    site_id = "gptoss"

    def start_new_question(self, page):
        """The playground offers gpt-oss-120b and gpt-oss-20b; the study
        wants 120b, so any visible 120b choice is clicked after loading a
        fresh page (the picker may already remember the last choice)."""
        super().start_new_question(page)
        # Only exact matches on real controls are clicked - the page also
        # shows download commands mentioning 120b that must not be touched.
        try:
            for element in page.query_selector_all("button, [role='option'], [role='tab'], label"):
                try:
                    if not element.is_visible():
                        continue
                    text = (element.inner_text() or "").strip().lower()
                    if text in ("gpt-oss-120b", "120b"):
                        element.click()
                        break
                except Exception:
                    continue
        except Exception:
            pass


DRIVER_CLASSES = {
    "openevidence": OpenEvidenceDriver,
    "uptodate": UpToDateDriver,
    "doximity": DoximityDriver,
    "chatgptclinicians": ChatGPTCliniciansDriver,
    "amboss": AmbossDriver,
    "clinicalkeyai": ClinicalKeyAIDriver,
    "dynamed": DynaMedDriver,
    "glasshealth": GlassHealthDriver,
    "gptoss": GptOssDriver,
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
