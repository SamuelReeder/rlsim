"""Opt-in recommendation review/apply workflow on a static site."""
import os
from pathlib import Path
import subprocess

import pytest
from test_browser_local import ROOT, static_url
from test_rig_ui import page, import_visual

pytestmark = pytest.mark.skipif(
    not os.getenv("MORPHOLAB_BROWSER_TEST"), reason="Opt-in training adviser UI"
)


def ready(page):
    page.wait_for_function("!state.previewPending && !state.initializing && !state.busy")


def test_template_velocity_suggestion_accepts_speed_precision(page):
    page.locator("[data-tab=train]").click()
    ready(page)
    before = page.evaluate("adviceSnapshot()")
    page.locator("#recommend-training").click()
    page.wait_for_function("state.advice && !state.adviceLoading")
    speed = page.evaluate("state.advice.settings.target_speed")
    # The default template's suggestion is not a multiple of 0.1 m/s.
    assert speed * 10 != round(speed * 10)
    page.locator("#apply-advice").click()
    assert page.evaluate("state.advice.applied")
    assert float(page.locator("#target-speed").input_value()) == speed
    assert page.evaluate("adviceFields.every(([id])=>document.getElementById(id).checkValidity())")
    page.locator("#undo-advice").click()
    assert page.evaluate("adviceSnapshot()") == before
    # Saved configurations copy this speed into the inference controls too.
    page.evaluate("speed=>document.querySelector('#eval-speed').value=speed", speed)
    assert page.locator("#eval-speed").evaluate("el=>el.checkValidity()")


def test_recommendations_review_apply_undo_and_preserve_custom_objective(page):
    page.locator("[data-tab=train]").click()
    ready(page)
    page.locator("#seed").fill("123")
    page.locator("#task-settings > summary").click()
    page.locator("#task-preset").select_option("custom")
    page.locator("#reward-expression").fill("upright + alive - 0.1*abs(vz)")
    ready(page)
    before = page.evaluate("({settings:adviceSnapshot(),task:readTask(),body:state.body,seed:document.querySelector('#seed').value})")
    page.locator("#recommend-training").click()
    page.wait_for_function("state.advice && !state.adviceLoading")
    assert page.locator("#training-advice").is_visible()
    assert page.evaluate("adviceSnapshot()") == before["settings"]
    assert page.locator("#advice-settings tr").count() >= 10
    assert "target_speed" not in page.evaluate("state.advice.settings")
    assert page.evaluate("state.status.status") == "idle"
    page.locator("#apply-advice").click()
    assert page.evaluate("state.advice.applied")
    assert page.evaluate("readTask()") == before["task"]
    assert page.evaluate("state.body") == before["body"]
    assert page.locator("#seed").input_value() == "123"
    assert page.evaluate("state.status.status") == "idle"  # Applying does not start a job.
    assert page.evaluate("adviceFields.every(([id])=>document.getElementById(id).checkValidity())")
    page.locator("#undo-advice").click()
    assert page.evaluate("adviceSnapshot()") == before["settings"]
    page.locator("#apply-advice").click()
    page.locator("#training-budget").fill("8192")
    assert page.locator("#training-advice").is_hidden()
    assert "Recommend again" in page.locator("#advice-status").inner_text()
    assert not page.evaluate("state.adviceUndo")
    page.locator("#recommend-training").click()
    page.wait_for_function("state.advice && !state.adviceLoading")
    page.locator("#friction").fill("0.1")
    assert page.locator("#training-advice").is_hidden()


def test_complex_body_recommendation_compiles_and_runs_short_ppo_check(page, tmp_path):
    supplied = os.getenv("RLSIM_FEMALE_ASSET")
    character = Path(supplied) if supplied else tmp_path / "humanoid.glb"
    if not supplied:
        subprocess.run([
            "node", "--input-type=module", "-e",
            "import {helperHumanoidGLB} from './tests/scene-fixtures.mjs';"
            "import {writeFileSync} from 'node:fs';"
            "writeFileSync(process.argv[1],helperHumanoidGLB());", str(character),
        ], cwd=ROOT, check=True)
    import_visual(page, character)
    original_xml = page.evaluate("state.body.xml")
    page.locator("[data-tab=train]").click()
    ready(page)
    page.locator("#recommend-training").click()
    page.wait_for_function("state.advice && !state.adviceLoading")
    recommendation = page.evaluate("state.advice.settings")
    assert recommendation["hidden_size"] >= 128
    assert recommendation["total_steps"] >= 524288
    assert recommendation["total_steps"] <= 2000000
    page.locator("#apply-advice").click()
    assert page.evaluate("state.body.xml") == original_xml
    assert page.evaluate("state.status.status") == "idle"
    output = ROOT / "test-results" / "advice"
    output.mkdir(parents=True, exist_ok=True)
    for width, height in [(1440, 1000), (390, 844)]:
        page.set_viewport_size({"width": width, "height": height})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth+1")
        page.locator("#training-advice > summary").scroll_into_view_if_needed()
        page.screenshot(path=str(output / f"recommendation-{width}.png"), full_page=True)
    # A short, deliberately capped check validates runtime compatibility only.
    # It does not establish that the full recommended budget will learn walking.
    page.evaluate("""async settings => browserBackend.request('/api/train', {
      ...settings,body:state.body,environment:readEnvironment(),
      task:{preset:'velocity',episode_seconds:1,terminate_on_fall:false},
      total_steps:4096,seed:17
    })""", recommendation)
    page.wait_for_function("""['Completed','Failed'].includes(
      document.querySelector('#run-status').textContent)""", timeout=180000)
    status = page.evaluate("browserBackend.request('/api/status')")
    assert status["status"] == "completed", status.get("error")
    saved = page.evaluate("id=>browserBackend.records(id)", status["run_id"])
    for key, value in recommendation.items():
        if key != "total_steps":
            assert saved["config"][key] == value
    assert saved["config"]["body"]["xml"] == original_xml
    assert saved["steps"] == 4096
    page.locator("[data-tab=run]").click()
    page.wait_for_function("state.selectedRun && !state.busy")
    assert float(page.locator("#eval-speed").input_value()) == recommendation["target_speed"]
    page.locator("#eval-speed").fill("0.91")
    page.evaluate("refreshRuns()")
    assert float(page.locator("#eval-speed").input_value()) == .91
    page.locator("#saved-run").select_option("")
    ready(page)
    page.locator("#saved-run").select_option(status["run_id"])
    ready(page)
    assert float(page.locator("#eval-speed").input_value()) == recommendation["target_speed"]


def test_unactuated_body_explains_why_settings_cannot_help(page):
    page.locator("#asset-files").set_input_files({
        "name": "passive.xml", "mimeType": "application/xml",
        "buffer": b'<mujoco><worldbody><body name="root" pos="0 0 1"><freejoint/><geom type="box" size=".1 .1 .2" mass="2"/></body></worldbody></mujoco>',
    })
    page.wait_for_function("!document.querySelector('#body-import-options').hidden")
    page.locator("#import-auto-actuators").uncheck()
    page.locator("#apply-body-import").click()
    page.wait_for_function("document.querySelector('#status-text').textContent==='Body imported on This device'")
    page.locator("[data-tab=train]").click()
    ready(page)
    page.locator("#recommend-training").click()
    page.wait_for_function("state.advice && !state.adviceLoading")
    assert page.evaluate("state.advice.settings") == {}
    assert page.locator("#apply-advice").is_hidden()
    assert "actuator" in page.locator("#advice-warnings").inner_text().lower()
