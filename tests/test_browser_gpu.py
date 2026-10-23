"""Opt-in real hardware WebGPU regression, using an isolated existing CDP browser.

MORPHOLAB_GPU_CDP=http://127.0.0.1:9339 MORPHOLAB_TEST_URL=http://localhost:8777 \
  .venv/bin/pytest -q -s tests/test_browser_gpu.py

Start the CDP browser with a temporary profile and normal GPU security settings.
This test does not start browsers, enable unsafe adapters, or touch a user profile.
"""
import json
import os
import time

import pytest

CDP = os.getenv("MORPHOLAB_GPU_CDP")
BASE = os.getenv("MORPHOLAB_TEST_URL", "http://localhost:8777").rstrip("/") + "/"
pytestmark = pytest.mark.skipif(not CDP, reason="Set MORPHOLAB_GPU_CDP for hardware WebGPU validation")


def test_real_webgpu_learning_and_worker_workflow():
    pw = pytest.importorskip("playwright.sync_api")
    with pw.sync_playwright() as runtime:
        browser = runtime.chromium.connect_over_cdp(CDP)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        # Keep the app's worker idle while checking independent policy kernels.
        page.goto(BASE + "?engine=server")
        page.add_script_tag(url=BASE + "vendor/tf.min.js")
        page.add_script_tag(url=BASE + "vendor/tf-backend-webgpu.min.js")
        result = page.evaluate("""async (base) => {
          const {PPO, selectBackend} = await import(base + "browser-policy.mjs");
          const hardware = await selectBackend();
          if (hardware.training_device !== "webgpu") return {hardware};
          const policy = new PPO(2, 1, {hidden_size:16, epochs:4, batch_size:64,
            seed:42, learning_rate:.003, entropy_coef:0});
          const observations = Array.from({length:128}, () => [1,.25]);
          const before = (await policy.act([[1,.25]], true)).actions[0][0];
          for (let i=0; i<12; i++) {
            const rollout = await policy.act(observations);
            const rewards = rollout.actions.map(action => -((action[0]-.8)**2));
            await policy.update({observations, actions:rollout.actions,
              oldLogProbs:rollout.logProbs, returns:rewards,
              advantages:rewards.map((reward,j) => reward-rollout.values[j])});
          }
          const after = await policy.act([[1,.25]], true);
          const restored = await PPO.restore(JSON.parse(JSON.stringify(await policy.serialize())));
          const same = JSON.stringify(await restored.act([[1,.25]], true)) === JSON.stringify(after);
          policy.dispose(); restored.dispose();
          return {hardware, before, after:after.actions[0][0], same, tensors:tf.memory().numTensors};
        }""", BASE)
        assert result["hardware"]["training_device"] == "webgpu", result
        assert abs(result["after"] - .8) < abs(result["before"] - .8) * .4, result
        assert result["same"] and result["tensors"] == 0, result
        # selectBackend itself also checks a 64-neuron, 128-sample PPO update
        # against CPU weights within 2e-5, catching invalid GPU command buffers.
        page.goto(BASE)
        errors, requests = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: requests.append(request.url) if "/api/" in request.url else None)
        page.wait_for_function("() => document.querySelector('#arena-empty').hidden", timeout=90000)
        assert page.locator("#hardware-label").inner_text() == "WebGPU · This device"
        page.locator("[data-tab=train]").click()
        page.locator("#advanced-training summary").click()
        page.locator("#hidden-size").fill("32")
        page.locator("#hidden-layers").fill("4")
        page.locator("#epochs").fill("2")
        page.locator("#rollout-steps").fill("64")
        page.locator("#training-budget").fill("4096")
        page.locator("#n-envs").fill("4")
        started = time.monotonic()
        page.locator("#train-button").click()
        page.wait_for_function("() => ['Completed','Failed'].includes(document.querySelector('#run-status').textContent)", timeout=180000)
        assert page.locator("#run-status").inner_text() == "Completed", page.locator("#status-text").inner_text()
        elapsed = time.monotonic() - started
        page.locator("[data-tab=run]").click()
        page.wait_for_function("() => document.querySelector('#saved-run').value")
        # Read the durable record directly: CDP may connect from WSL/Linux to a
        # Windows browser, whose download paths are not local to this test host.
        record = page.evaluate("""async () => {
          const request = indexedDB.open("morpholab-browser-v1", 1);
          const db = await new Promise((resolve,reject) => {
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
          });
          const lookup = db.transaction("runs").objectStore("runs")
            .get(document.querySelector("#saved-run").value);
          return await new Promise((resolve,reject) => {
            lookup.onsuccess = () => { db.close(); resolve(lookup.result); };
            lookup.onerror = () => { db.close(); reject(lookup.error); };
          });
        }""")
        assert record["steps"] == 4096 and record["hardware"]["training_device"] == "webgpu"
        assert record["policy"]["weights"]["actor_w4"]["shape"] == [32, 32]
        assert record["config"]["hidden_layers"] == 4
        assert len(record["history"]) >= 8 and record["metrics"]["loss"] is not None
        page.locator("#replay-button").click()
        page.wait_for_function("() => document.querySelector('#viewer-mode').textContent === 'Model inference'", timeout=60000)
        page.wait_for_function("!document.querySelector('#evaluation-summary').hidden", timeout=60000)
        assert float(page.locator("#timeline").get_attribute("max")) > 0
        assert not requests and not errors, (requests, errors)
        print(json.dumps({**result, "worker_steps": record["steps"], "worker_seconds": round(elapsed, 2),
                          "worker_fps": record["metrics"]["fps"], "replay": "passed"}))
        context.close()
        browser.close()


@pytest.mark.parametrize("control_mode",["torque","joint_target"])
def test_imported_character_trains_and_replays_on_webgpu(control_mode):
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    rig = subprocess.check_output(['node', '--input-type=module', '-e',
        "import {tinyRigGLB} from './tests/scene-fixtures.mjs'; process.stdout.write(Buffer.from(tinyRigGLB(6)).toString('base64'));"], cwd=root, text=True)
    pw = pytest.importorskip('playwright.sync_api')
    with pw.sync_playwright() as runtime:
        browser = runtime.chromium.connect_over_cdp(CDP)
        context = browser.new_context()
        try:
            page = context.new_page()
            page.goto(BASE)
            page.wait_for_function("document.querySelector('#arena-empty').hidden", timeout=90000)
            info = page.evaluate('''async ({data,controlMode}) => {
              const bytes=Uint8Array.from(atob(data),c=>c.charCodeAt(0));
              const bundle=await browserBackend.prepareBodyFiles([new File([bytes],'character.glb')]);
              let imported=await browserBackend.importBody(bundle,{entry:'character.glb'});
              if(controlMode==='joint_target'){
                const rig=structuredClone(imported.body.physical_rig);
                rig.control={mode:'joint_target',excursion:30,stiffness:1,damping_ratio:.7};
                const body=RLsimScene.rebuildPhysicalRig(imported.body,rig);
                imported=await browserBackend.call('preview',{body,environment:{gravity:9.81,friction:1}});
              }
              const state=await browserBackend.call('train',{body:imported.body,
                environment:{gravity:9.81,friction:1},total_steps:4096,n_envs:2,
                hidden_size:32,hidden_layers:3,epochs:1,rollout_steps:64,
                task:{preset:'balance',episode_seconds:1,terminate_on_fall:false}});
              return imported.model_info;
            }''', {"data":rig,"controlMode":control_mode})
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                status = page.evaluate("async()=>browserBackend.call('status')")
                if status['status'] in ('completed','failed'):
                    break
                page.wait_for_timeout(250)
            assert status['status'] == 'completed', status.get('error')
            assert status['hardware']['training_device'] == 'webgpu'
            assert info['observation_schema'] == 'rlsim-general-v1'
            # Six preserved bones: a free root plus five 3-axis joints.
            assert info['action_size'] == 15
            assert info['action_schema'] == ("rlsim-joint-target-v1" if control_mode=="joint_target" else "rlsim-actuator-controls-v1")
            replay = page.evaluate("async id=>browserBackend.request('/api/replay',{run_id:id})", status['run_id'])
            warm = page.evaluate("async id=>browserBackend.request('/api/replay',{run_id:id})", status['run_id'])
            assert replay["checkpoint"] == "best"
            assert warm["inference_cache_hit"] is True
            assert warm["frames"] == replay["frames"]
            assert replay['frames'][-1]['time'] == pytest.approx(1, abs=.021)
            assert replay['body']['visual']['mode'] == 'skinned'
            print(json.dumps({'gpu_character_steps':status['steps'],'seconds':status['elapsed'],
                'observations':info['observation_size'],'actions':info['action_size'],'control_mode':control_mode}))
        finally:
            context.close()
