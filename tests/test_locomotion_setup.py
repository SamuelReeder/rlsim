"""Opt-in locomotion physics review/apply/undo on an isolated static site.

MORPHOLAB_BROWSER_TEST=1 .venv/bin/pytest -q tests/test_locomotion_setup.py
Optional: RLSIM_VALERY_ASSET=/absolute/path/to/a/rigged/character.glb
The capped PPO check proves runtime compatibility, not learned walking.
"""
import math
import os
from pathlib import Path
import re
import subprocess

import pytest
from test_browser_local import ROOT, static_url
from test_rig_ui import import_visual, page

pytestmark = pytest.mark.skipif(
    not os.getenv("MORPHOLAB_BROWSER_TEST"), reason="Opt-in locomotion setup UI"
)


def ready(page):
    page.wait_for_function("!state.previewPending && !state.initializing && !state.busy")


@pytest.fixture
def character(tmp_path):
    supplied = os.getenv("RLSIM_VALERY_ASSET")
    if supplied:
        asset = Path(supplied)
        assert asset.is_file(), f"Missing requested character: {asset}"
        return asset
    asset = tmp_path / "humanoid-fingers.glb"
    # Extend the original, locally generated humanoid fixture with weighted hand
    # and finger bones. No downloaded asset is needed for these regression tests.
    subprocess.run([
        "node", "--input-type=module", "-e", """
        import {readFileSync, writeFileSync} from 'node:fs';
        const source=readFileSync('./tests/scene-fixtures.mjs','utf8');
        const anchor="['Spine',1,[0,1.3,0]],['Head',8,[0,1.6,0]],['PoleTargetL',0,[.2,.5,.8]],";
        if(!source.includes(anchor))throw new Error('Humanoid fixture anchor changed');
        const extra=`
          ['ShoulderL',8,[.15,1.38,0]],['UpperArmL',11,[.3,1.38,0]],
          ['LowerArmL',12,[.55,1.38,0]],['HandL',13,[.75,1.38,0]],
          ['IndexFinger1L',14,[.82,1.38,0]],['IndexFinger2L',15,[.86,1.38,0]],
          ['ShoulderR',8,[-.15,1.38,0]],['UpperArmR',17,[-.3,1.38,0]],
          ['LowerArmR',18,[-.55,1.38,0]],['HandR',19,[-.75,1.38,0]],
          ['IndexFinger1R',20,[-.82,1.38,0]],['IndexFinger2R',21,[-.86,1.38,0]],`;
        const module=await import('data:text/javascript;base64,'+
          Buffer.from(source.replace(anchor,anchor+extra)).toString('base64'));
        writeFileSync(process.argv[1],module.helperHumanoidGLB());
        """, str(asset),
    ], cwd=ROOT, check=True)
    return asset


def setup_review(page, character):
    import_visual(page, character)
    ready(page)
    original = page.evaluate("structuredClone(state.body)")
    assert page.locator("#locomotion-setup").is_visible()
    if not page.locator("#locomotion-setup").evaluate("el=>el.open"):
        page.locator("#locomotion-setup > summary").click()
    page.locator("#locomotion-mass").fill("60")
    page.locator("#locomotion-strength").fill("1")
    page.locator("#locomotion-lock-fingers").check()
    page.locator("#recommend-locomotion").click()
    page.wait_for_function("state.locomotionAdvice && !state.busy")
    assert page.locator("#locomotion-review").is_visible()
    assert page.locator("#apply-locomotion").is_enabled()
    assert page.evaluate("state.body") == original
    assert page.evaluate("state.status.status") == "idle"


def finger_parts(body):
    return [
        part for part in body["physical_rig"]["bodies"]
        if re.search(r"finger|thumb|index|middle|pinky|ring", part["label"], re.I)
    ]


def assert_finite(value):
    if isinstance(value, float):
        assert math.isfinite(value), value
    elif isinstance(value, dict):
        for child in value.values():
            assert_finite(child)
    elif isinstance(value, list):
        for child in value:
            assert_finite(child)


def test_locomotion_review_apply_undo_preserves_visual_and_saved_model(page, character):
    setup_review(page, character)
    original = page.evaluate("structuredClone(state.body)")
    original_info = page.evaluate("structuredClone(state.modelInfo)")
    controls_before = original_info["action_size"]
    original_settings = page.evaluate("({task:readTask(),environment:readEnvironment(),training:adviceSnapshot()})")
    fingers = finger_parts(original)
    assert fingers, "The character fixture must contain fingers"
    assert any(part["joint"]["type"] != "fixed" for part in fingers)
    assert page.evaluate("state.status.status") == "idle"
    # Merely requesting a recommendation must leave the current body untouched.
    assert page.evaluate("state.body") == original
    page.evaluate("""() => {
      const saved={id:'previous-controller',config:{body:structuredClone(state.body)}};
      state.runs.push(saved);state.selectedRun=saved.id;
      document.querySelector('#saved-run').add(new Option('Previous controller',saved.id));
      document.querySelector('#saved-run').value=saved.id;
      controls();
    }""")
    page.locator("#apply-locomotion").click()
    page.wait_for_function("state.locomotionUndo && !state.busy", timeout=60000)
    revised = page.evaluate("structuredClone(state.body)")
    assert sum(part["mass"] for part in revised["physical_rig"]["bodies"]) == pytest.approx(60)
    assert revised["xml"] != original["xml"]
    assert revised["visual"]["files"] == original["visual"]["files"]
    assert [part["source_bones"] for part in revised["physical_rig"]["bodies"]] == [
        part["source_bones"] for part in original["physical_rig"]["bodies"]
    ]
    revised_by_id = {part["id"]: part for part in revised["physical_rig"]["bodies"]}
    for original_part in fingers:
        part = revised_by_id[original_part["id"]]
        assert part["joint"]["type"] == "fixed"
        assert f'joint="joint_{part["id"]}"' not in revised["xml"]
        assert f'name="joint_{part["id"]}"' not in revised["xml"]
        assert part["position"] == original_part["position"]
        assert part["quaternion"] == original_part["quaternion"]
        assert part["colliders"] == original_part["colliders"]
    assert page.evaluate("state.modelInfo.action_size") < controls_before
    assert not page.evaluate("state.selectedRun")
    assert page.evaluate("state.runs.find(run=>run.id==='previous-controller').config.body") == original
    assert page.evaluate("({task:readTask(),environment:readEnvironment(),training:adviceSnapshot()})") == original_settings
    assert page.evaluate("state.status.status") == "idle"
    assert page.locator("#undo-locomotion").is_enabled()
    output = ROOT / "test-results" / "locomotion"
    output.mkdir(parents=True, exist_ok=True)
    for width, height in [(1440, 1000), (390, 844)]:
        page.set_viewport_size({"width": width, "height": height})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth+1")
        page.locator("#locomotion-setup").scroll_into_view_if_needed()
        page.screenshot(path=str(output / f"locomotion-{width}.png"), full_page=True)
    page.locator("#undo-locomotion").click()
    page.wait_for_function("!state.locomotionUndo && !state.busy", timeout=60000)
    assert page.evaluate("state.body") == original
    assert page.evaluate("state.modelInfo.action_size") == controls_before
    assert not page.evaluate("state.selectedRun")
    assert page.evaluate("state.runs.find(run=>run.id==='previous-controller').config.body") == original


def test_locomotion_review_invalidates_when_settings_change_and_manual_joint_lock(page, character):
    setup_review(page, character)
    original = page.evaluate("structuredClone(state.body)")
    page.locator("#locomotion-mass").fill("65")
    assert not page.evaluate("state.locomotionAdvice")
    assert page.locator("#apply-locomotion").is_hidden() or page.locator("#apply-locomotion").is_disabled()
    assert page.evaluate("state.body") == original
    page.locator("#recommend-locomotion").click()
    page.wait_for_function("state.locomotionAdvice && !state.busy")
    page.locator("#locomotion-lock-fingers").uncheck()
    assert not page.evaluate("state.locomotionAdvice")
    page.locator("#physical-rig-editor > summary").click()
    part = next(part for part in finger_parts(original) if part["joint"]["type"] != "fixed")
    page.locator("#rig-body").select_option(part["id"])
    page.locator("#rig-joint").select_option("fixed")
    assert page.evaluate("state.rigDirty")
    assert page.evaluate("state.body") == original
    page.locator("#apply-physical-rig").click()
    page.wait_for_function("document.querySelector('#status-text').textContent.startsWith('Physics applied')", timeout=60000)
    locked = page.evaluate("id=>state.body.physical_rig.bodies.find(part=>part.id===id)", part["id"])
    assert locked["joint"]["type"] == "fixed"
    assert locked["position"] == part["position"]
    assert locked["quaternion"] == part["quaternion"]
    assert locked["joint"]["torque"] == part["joint"]["torque"]
    assert page.evaluate("state.body.visual.files") == original["visual"]["files"]
    # Switching the same part back restores its original movable type and fields.
    page.locator("#rig-joint").select_option(part["joint"]["type"])
    page.locator("#apply-physical-rig").click()
    page.wait_for_function("document.querySelector('#status-text').textContent.startsWith('Physics applied') && !state.busy", timeout=60000)
    restored = page.evaluate("id=>state.body.physical_rig.bodies.find(part=>part.id===id)", part["id"])
    assert restored == part


def test_locomotion_body_short_ppo_training_and_replay_remain_finite(page, character):
    setup_review(page, character)
    page.locator("#apply-locomotion").click()
    page.wait_for_function("state.locomotionUndo && !state.busy", timeout=60000)
    body = page.evaluate("structuredClone(state.body)")
    page.evaluate("""async () => browserBackend.request('/api/train',{
      body:state.body,environment:readEnvironment(),
      task:{preset:'velocity',episode_seconds:1,terminate_on_fall:false},
      target_speed:.4,total_steps:4096,n_envs:2,hidden_size:32,hidden_layers:2,
      rollout_steps:128,batch_size:128,epochs:2,learning_rate:.0003,
      curriculum:false,randomize:false,seed:31
    })""")
    page.wait_for_function("""['Completed','Failed'].includes(
      document.querySelector('#run-status').textContent)""", timeout=180000)
    status = page.evaluate("browserBackend.request('/api/status')")
    assert status["status"] == "completed", status.get("error")
    saved = page.evaluate("id=>browserBackend.records(id)", status["run_id"])
    assert saved["config"]["body"] == body
    assert saved["steps"] == 4096
    assert_finite(saved["history"])
    assert_finite(saved["metrics"])
    # The saved controller is evaluated with the body it was actually trained on.
    replay = page.evaluate("""async id => browserBackend.request('/api/replay',{
      run_id:id,environment:readEnvironment(),target_speed:.4,seed:237
    })""", status["run_id"])
    assert replay["source"] == "trained"
    assert replay["body"] == body
    assert replay["summary"]["steps"] > 0
    assert len(replay["frames"]) > 1
    assert_finite(replay["frames"])
    assert_finite(replay["summary"])
