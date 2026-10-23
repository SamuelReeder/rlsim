"""Real worker cancellation of trained predictions, loading, and baseline simulation."""
import os
from pathlib import Path

import pytest
from test_browser_local import static_url

pytestmark = pytest.mark.skipif(not os.getenv("MORPHOLAB_BROWSER_TEST"), reason="Opt-in browser cancellation")


def test_stop_inference_cancels_generation_and_retains_partial_playback(static_url):
    pw = pytest.importorskip("playwright.sync_api")
    with pw.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900})

        def slow_predictions(route):
            response = route.fetch()
            source = response.text()
            needle = "const inputs=model?[model.normalizer.normalize(obs)]:null;"
            assert source.count(needle) == 1
            source = source.replace(needle, """
              await new Promise(resolve=>setTimeout(resolve,5));
              if(cancelled())break;
              globalThis.testPredictions=(globalThis.testPredictions||0)+1;
            """ + needle)
            route.fulfill(response=response, body="Object.defineProperty(navigator,'gpu',{value:undefined});\n" + source)

        context.route("**/browser-worker.js", slow_predictions)
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(static_url)
        page.wait_for_function("state.connected && !state.previewPending && !state.initializing", timeout=90000)
        page.evaluate("""() => {
          selectTab('train',false);
          setTask({preset:'velocity',episode_seconds:1,terminate_on_fall:false});
          setAdvanced({hidden_size:32,epochs:1,best_checkpoint:false});
          $('training-budget').value=4096;$('n-envs').value=2;
          return train();
        }""")
        page.wait_for_function("['completed','failed'].includes(state.status.status)", timeout=120000)
        assert page.evaluate("state.status.status") == "completed"
        page.wait_for_function("!!state.selectedRun")
        page.evaluate("""async () => {
          const record=await browserBackend.records(state.selectedRun);
          record.config.task.episode_seconds=120;
          record.config.task.terminate_on_fall=false;
          await browserBackend.put(record);await refreshRuns(record.id);
          selectTab('run',false);
        }""")
        page.locator("#replay-button").click()
        page.wait_for_function("state.replayLoading && state.frames.length>4")
        assert page.locator("#stop-inference").inner_text() == "Stop inference"
        page.locator("#play-pause").click()
        assert not page.evaluate("state.playing")
        before = page.workers[0].evaluate("testPredictions")
        page.wait_for_function("state.frames.at(-1).time>.6")
        assert page.workers[0].evaluate("testPredictions") > before, "Pause only pauses playback."
        output = Path(__file__).resolve().parents[1] / "test-results" / "stop-inference"
        output.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(output / "desktop.png"))
        page.set_viewport_size({"width":390,"height":844})
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth+1")
        page.screenshot(path=str(output / "mobile.png"))
        page.locator("#stop-inference").click()
        page.wait_for_function("!state.busy && !state.replayLoading", timeout=10000)
        assert page.locator("#status-text").inner_text() == "Inference stopped"
        assert "Partial episode" in page.locator("#evaluation-summary").inner_text()
        assert page.locator("#stop-inference").is_hidden()
        assert not page.evaluate("state.playing")
        after = page.workers[0].evaluate("testPredictions")
        frames = page.evaluate("state.frames.length")
        assert 1 < frames < 3000
        page.locator("#play-pause").click()
        page.wait_for_timeout(250)
        assert page.workers[0].evaluate("testPredictions") == after
        assert page.evaluate("state.frames.length") == frames
        assert page.locator("#replay-button").is_enabled()

        # Stop during IndexedDB loading, before the worker receives the request.
        page.evaluate("""() => {
          const original=browserBackend.records.bind(browserBackend);
          browserBackend.records=async id=>{
            await new Promise(resolve=>window.releaseModelRead=resolve);
            return original(id);
          };
          window.restoreModelRead=()=>browserBackend.records=original;
          window.loadingReplay=infer();
        }""")
        page.locator("#stop-inference").click()
        page.evaluate("releaseModelRead();restoreModelRead()")
        page.wait_for_function("!state.busy")
        assert page.locator("#status-text").inner_text() == "Inference stopped"
        assert page.workers[0].evaluate("testPredictions") == after

        # A fresh inference still works after cancellation.
        page.locator("#replay-button").click()
        page.wait_for_function("state.frames.length>4 && state.replayLoading")
        page.locator("#stop-inference").click()
        page.wait_for_function("!state.busy")
        assert page.workers[0].evaluate("testPredictions") > after
        page.evaluate("""() => {
          selectTab('body',false);
          setTask({preset:'balance',episode_seconds:120,terminate_on_fall:false});
          window.baselineReplay=infer(true);
        }""")
        page.wait_for_function("state.frames.length>4 && state.replayLoading")
        assert page.locator("#stop-inference").inner_text() == "Stop simulation"
        page.locator("#stop-inference").click()
        page.wait_for_function("!state.busy")
        assert page.locator("#status-text").inner_text() == "Simulation stopped"
        assert not errors, errors
        context.close()
        browser.close()
