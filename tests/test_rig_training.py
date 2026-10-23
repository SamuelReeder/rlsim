"""Opt-in real PPO/checkpoint roundtrip for a generated physical rig.

This is a workflow check, not a locomotion benchmark. Set RLSIM_FEMALE_ASSET
to exercise a real local character instead of the small humanoid fixture.
"""
import os
import json
from pathlib import Path
import subprocess

import pytest
from test_browser_local import ROOT, static_url
from test_rig_ui import page, import_visual

pytestmark = pytest.mark.skipif(
    not os.getenv("MORPHOLAB_BROWSER_TEST"), reason="Opt-in generated-rig training"
)


def test_generated_rig_trains_saves_and_replays_its_own_physics(page, tmp_path):
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
    original = page.evaluate("structuredClone(state.body)")
    assert original["physical_rig"]["profile"] == "humanoid"
    page.evaluate("""async () => browserBackend.request('/api/train', {
      body:state.body, environment:readEnvironment(),
      task:{preset:'balance',episode_seconds:1,terminate_on_fall:false},
      total_steps:4096,n_envs:2,target_speed:.5,seed:17,
      hidden_size:16,hidden_layers:1,batch_size:64,epochs:1,rollout_steps:64,
      randomize:false,curriculum:false
    })""")
    page.wait_for_function("""() =>
      ['Completed','Failed'].includes(document.querySelector('#run-status').textContent)
    """, timeout=180000)
    status = page.evaluate("browserBackend.request('/api/status')")
    assert status["status"] == "completed", status.get("error")
    run_id = status["run_id"]
    record = page.evaluate("id => browserBackend.records(id)", run_id)
    assert record["steps"] == 4096
    assert record["normalizer"]["count"] > 4096
    assert record["config"]["body"]["xml"] == original["xml"]
    assert record["config"]["body"]["physical_rig"]["root"] == original["physical_rig"]["root"]
    assert record["policy"]["actionSize"] == record["model_info"]["action_size"]

    # A fresh worker must restore the saved graph, normalizer, and policy together.
    page.reload()
    page.wait_for_function("document.querySelector('#arena-empty').hidden", timeout=90000)
    replay = page.evaluate(
        "id => browserBackend.request('/api/replay',{run_id:id,seed:10007})", run_id
    )
    assert replay["body"]["xml"] == original["xml"]
    assert replay["body"]["physical_rig"]["root"] == original["root_body"]
    assert replay["frames"][-1]["time"] == pytest.approx(1, abs=.021)
    assert replay["model_info"]["tracked_body"] == original["root_body"]

    streamed = page.evaluate("""async id => {
      const reference=await browserBackend.request('/api/replay',{run_id:id,seed:10007});
      const started=performance.now(),events=[];
      const result=await browserBackend.request('/api/replay',{run_id:id,seed:10007},{
        onProgress:event=>events.push({phase:event.phase,ms:performance.now()-started})
      });
      return {
        first_frame_ms:events.find(event=>event.phase==='ready').ms,
        complete_ms:performance.now()-started,
        same_frames:JSON.stringify(result.frames)===JSON.stringify(reference.frames),
        same_summary:JSON.stringify(result.summary)===JSON.stringify(reference.summary)
      };
    }""", run_id)
    assert streamed["same_frames"] and streamed["same_summary"]
    assert streamed["first_frame_ms"] < streamed["complete_ms"]
    print(json.dumps({"rig_replay": streamed}))
