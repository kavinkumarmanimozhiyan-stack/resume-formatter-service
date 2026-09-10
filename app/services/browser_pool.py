"""
browser_pool.py
─────────────────────────────────────────────────────────────────────────────
One persistent Chromium instance, reused across requests, instead of
launching/tearing down a fresh browser process per PDF render.

Playwright's sync API is bound to the thread that created it, so the
browser lives on a single dedicated worker thread (ThreadPoolExecutor
with max_workers=1 always reuses that same thread). Every render request
is submitted to that thread; only a `browser.new_context()` is created
and torn down per call, not the browser process itself.
"""

import atexit
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pw-render")
_state = threading.local()


def _get_browser():
    if not hasattr(_state, "browser"):
        from playwright.sync_api import sync_playwright
        _state.playwright = sync_playwright().start()
        _state.browser = _state.playwright.chromium.launch()
        print("[BROWSER_POOL] Chromium launched (persistent)")
    return _state.browser


def _render(html: str, pdf_kwargs: dict) -> bytes:
    browser = _get_browser()
    context = browser.new_context()
    try:
        page = context.new_page()
        page.set_content(html, wait_until="networkidle")
        page.emulate_media(media="print")
        return page.pdf(**pdf_kwargs)
    finally:
        context.close()


def render_pdf(html: str, pdf_kwargs: dict[str, Any], timeout: float = 60) -> bytes:
    """Render HTML to PDF bytes on the persistent-browser worker thread."""
    future = _executor.submit(_render, html, pdf_kwargs)
    return future.result(timeout=timeout)


def _shutdown() -> None:
    def _close():
        if hasattr(_state, "browser"):
            _state.browser.close()
            _state.playwright.stop()
    try:
        _executor.submit(_close).result(timeout=10)
    except Exception:
        pass
    _executor.shutdown(wait=False)


atexit.register(_shutdown)
