"""Exercise streaming playback controls under a real worker buffer underrun."""
import os
from pathlib import Path

import pytest
from test_browser_local import static_url

pytestmark = pytest.mark.skipif(
    not os.getenv("MORPHOLAB_BROWSER_TEST"), reason="Opt-in browser playback"
)


def test_streamed_playback_buffers_and_preserves_pause(static_url):
    pw = pytest.importorskip("playwright.sync_api")
    with pw.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900})

        def delayed_worker(route):
            response = route.fetch()
            source = response.text()
            # Hold one real replay midway to reproduce slow inference reliably.
            needle = "if(steps%16===0)await pause();"
            assert source.count(needle) == 1
            source = source.replace(needle, """
              if(input.stream && steps===128){
                flush();
                await new Promise(resolve=>{globalThis.resumeTestReplay=resolve;});
              }
            """ + needle)
            route.fulfill(response=response, body=(
                "Object.defineProperty(navigator,'gpu',{value:undefined});\n" + source
            ))

        context.route("**/browser-worker.js", delayed_worker)
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(static_url)
        page.wait_for_function("document.querySelector('#arena-empty').hidden", timeout=90000)
        page.evaluate("""() => {
          setTask({preset:'balance',episode_seconds:6,terminate_on_fall:false});
          window.replayCheck=infer(true);
        }""")
        page.wait_for_function("""state.replayLoading && state.frames.length>1 &&
          state.replayBuffering && state.frames.at(-1).time>2""", timeout=30000)
        assert page.locator("#replay-progress").is_visible()
        assert "Buffering" in page.locator("#replay-progress").inner_text()
        assert page.locator("#evaluation-summary").is_hidden()
        assert page.locator("#play-pause").is_enabled()
        output = Path(__file__).resolve().parents[1] / "test-results" / "playback"
        output.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(output / "buffering-desktop.png"))
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth+1")
        page.screenshot(path=str(output / "buffering-mobile.png"))
        before = page.evaluate("state.replayTime")
        page.locator("#play-pause").click()
        assert not page.evaluate("state.playing")
        assert page.evaluate("state.replayTime") == before  # Buffer end is not replay end.
        page.workers[0].evaluate("resumeTestReplay()")
        page.wait_for_function("!state.busy", timeout=30000)
        assert not page.evaluate("state.playing")  # Completion must not resume a paused replay.
        assert page.evaluate("state.replayTime") == before
        assert page.evaluate("state.frames.at(-1).time") == pytest.approx(6, abs=.021)
        assert page.locator("#replay-progress").is_hidden()
        assert not page.locator("#evaluation-summary").evaluate("element=>element.hidden")
        assert not errors, errors
        context.close()
        browser.close()
