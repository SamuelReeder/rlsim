"""Opt-in end-to-end checkpoint, curriculum, servo and cached replay checks."""
import os
import pytest
from test_browser_local import ROOT, static_url
from test_rig_ui import page
from test_locomotion_setup import character, setup_review, assert_finite

pytestmark = pytest.mark.skipif(not os.getenv("MORPHOLAB_BROWSER_TEST"), reason="Opt-in real browser training")


def test_servo_curriculum_best_latest_and_cached_replay(page, character):
    setup_review(page, character)
    page.locator("#apply-locomotion").click()
    page.wait_for_function("state.locomotionUndo && !state.busy", timeout=60000)
    visual = page.evaluate("state.body.visual")
    page.locator("#physical-rig-editor > summary").click()
    page.locator("#rig-control-mode").select_option("joint_target")
    page.locator("#apply-physical-rig").click()
    page.wait_for_function("document.querySelector('#status-text').textContent.startsWith('Physics applied') && !state.busy", timeout=60000)
    assert page.evaluate("state.modelInfo.action_schema") == "rlsim-joint-target-v1"
    assert page.locator("#baseline-button").inner_text() == "Simulate rest targets"
    assert page.evaluate("state.body.visual") == visual
    assert "<orientation" in page.evaluate("state.body.xml")
    page.locator("[data-tab=train]").click()
    page.wait_for_function("!state.previewPending")
    page.locator("#curriculum-mode").select_option("staged")
    page.locator("#advanced-training > summary").click()
    for key, value in {"training-budget":4096, "n-envs":2, "hidden-size":32,
                       "epochs":2, "evaluation-interval":4096, "evaluation-episodes":1}.items():
        page.locator(f"#{key}").fill(str(value))
    page.locator("#task-settings > summary").click()
    page.locator("#episode-seconds").fill("1")
    page.wait_for_function("!state.previewPending")
    page.locator("#train-button").click()
    page.wait_for_function("['Completed','Failed'].includes(document.querySelector('#run-status').textContent)", timeout=180000)
    assert page.locator("#run-status").inner_text() == "Completed", page.locator("#status-text").inner_text()
    status=page.evaluate("browserBackend.request('/api/status')")
    record=page.evaluate("id=>browserBackend.records(id)", status["run_id"])
    assert record["best"]["validation"]["episodes"]
    assert record["config"]["curriculum_mode"] == "staged"
    assert record["curriculum_state"]["name"] in ["Balance","Slow walk","Walk","Target speed"]
    assert record["policy"]["config"]["target_kl"] == .03
    assert record["best"]["normalizer"]["size"] == record["best"]["policy"]["obsSize"]
    assert_finite(record["best"]["validation"])
    page.locator("[data-tab=run]").click()
    page.wait_for_function("!state.previewPending && !!state.selectedRun")
    assert page.locator("#checkpoint-choice").input_value() == "best"
    assert "Best at" in page.locator("#checkpoint-status").inner_text()
    result=page.evaluate("""async id=>{
      const input={run_id:id,checkpoint:'best',seed:400007};
      const first=await browserBackend.request('/api/replay',input);
      const second=await browserBackend.request('/api/replay',input);
      const latest=await browserBackend.request('/api/replay',{...input,checkpoint:'latest'});
      return {first,second,latest};
    }""", status["run_id"])
    assert result["first"]["checkpoint"] == "best"
    assert result["first"]["inference_cache_hit"] is False
    assert result["second"]["inference_cache_hit"] is True
    assert result["first"]["frames"] == result["second"]["frames"]
    assert result["latest"]["checkpoint"] == "latest"
    assert result["latest"]["inference_cache_hit"] is False
    assert_finite(result)
    output=ROOT/"test-results"/"training-options"
    output.mkdir(parents=True, exist_ok=True)
    for width,height in [(1440,1000),(390,844)]:
        page.set_viewport_size({"width":width,"height":height})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth+1")
        page.screenshot(path=str(output/f"model-{width}.png"),full_page=True)
    page.reload()
    page.wait_for_function("state.connected && !state.initializing", timeout=60000)
    restored=page.evaluate("id=>browserBackend.records(id)",status["run_id"])
    assert restored["best"] == record["best"], "Both checkpoint variants persist across reload."
    fresh=page.evaluate("id=>browserBackend.request('/api/replay',{run_id:id,checkpoint:'best',seed:400007})",status["run_id"])
    assert fresh["inference_cache_hit"] is False
    assert fresh["frames"] == result["first"]["frames"], "Fresh-worker restore must reproduce cached replay."
    resumed=page.evaluate("id=>browserBackend.request('/api/resume',{run_id:id,additional_steps:4096})",status["run_id"])
    page.wait_for_function("id=>state.status.run_id===id&&['completed','failed'].includes(state.status.status)",arg=resumed["run_id"],timeout=180000)
    continued=page.evaluate("browserBackend.call('status')")
    assert continued['status']=='completed',continued.get('error')
    child=page.evaluate('id=>browserBackend.records(id)',resumed['run_id'])
    assert child['steps']==8192
    assert child['config']['body']==record['config']['body']
    assert child['continuation']['optimizer']['iterations']>record['continuation']['optimizer']['iterations']
    assert child['continuation']['curriculum']['index']>=record['continuation']['curriculum']['index']
    assert child['model_info']['action_schema']=='rlsim-joint-target-v1'


def test_training_and_servo_controls_fit_mobile_and_desktop(page, character):
    setup_review(page, character)
    page.locator("#physical-rig-editor > summary").click()
    page.locator("#rig-control-mode").select_option("joint_target")
    page.locator("#apply-physical-rig").click()
    page.wait_for_function("document.querySelector('#status-text').textContent.startsWith('Physics applied') && !state.busy", timeout=60000)
    assert page.locator("#baseline-button").inner_text() == "Simulate rest targets"
    output=ROOT/"test-results"/"training-options"
    output.mkdir(parents=True,exist_ok=True)
    for tab in ["body","train"]:
        page.locator(f"[data-tab={tab}]").click()
        page.wait_for_function("!state.previewPending")
        if tab=="train":
            page.locator("#advanced-training > summary").click()
            page.locator("#curriculum-mode").select_option("staged")
        for width,height in [(1440,1000),(390,844)]:
            page.set_viewport_size({"width":width,"height":height})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth+1")
            page.screenshot(path=str(output/f"{tab}-{width}.png"),full_page=True)
    response=page.request.get(page.url+"rig-help.html")
    assert response.ok
    assert 'id="training-options"' in response.text()
    assert 'id="joint-targets"' in response.text()
