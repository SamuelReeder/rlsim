"""Exercise coordinated training, portable import and inference in a real worker."""
import json
import os
from pathlib import Path
import pytest
from test_browser_local import static_url
pytestmark=pytest.mark.skipif(not os.getenv('MORPHOLAB_BROWSER_TEST'),reason='Opt-in browser training')


def test_walking_training_export_and_replay(static_url):
    pw=pytest.importorskip('playwright.sync_api')
    with pw.sync_playwright() as p:
        cdp=os.getenv("MORPHOLAB_GPU_CDP")
        browser=p.chromium.connect_over_cdp(cdp) if cdp else p.chromium.launch(headless=True)
        context=browser.new_context()
        def cpu(route):
            response=route.fetch();route.fulfill(response=response,body="Object.defineProperty(navigator,'gpu',{value:undefined});\n"+response.text())
        if not cdp:context.route('**/browser-worker.js',cpu)
        page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(os.getenv('MORPHOLAB_TEST_URL',static_url))
        page.wait_for_function('state.connected&&!state.initializing&&!state.previewPending',timeout=90000)
        if cdp:assert page.evaluate('state.hardware.training_device')=='webgpu'
        page.locator('[data-tab=train]').click()
        page.locator('#task-settings > summary').click();page.locator('#task-preset').select_option('walk')
        page.wait_for_function('state.modelInfo.observation_schema==="rlsim-planar-gait-v1"&&!state.previewPending')
        assert page.locator('#walking-help').is_visible()
        assert page.locator('#baseline-button').text_content()=='Simulate rest targets'
        assert page.locator('#foot-only-support').is_checked()
        page.locator('#mechanical-work-weight').fill('0.2')
        assert page.evaluate('state.modelInfo.action_schema')=='rlsim-joint-targets-v1'
        assert page.evaluate('state.modelInfo.observation_size')==57
        page.locator('#advanced-training > summary').click()
        for id,value in [('training-budget','4096'),('n-envs','2'),('hidden-size','32'),('epochs','1'),('rollout-steps','128'),('batch-size','128')]:page.locator('#'+id).fill(value)
        page.locator('#best-checkpoint').uncheck();page.locator('#train-button').click()
        page.wait_for_function("['Completed','Failed'].includes(document.querySelector('#run-status').textContent)",timeout=180000)
        assert page.locator('#run-status').inner_text()=='Completed',page.locator('#status-text').inner_text()
        page.locator('[data-tab=run]').click()
        page.wait_for_function('state.selectedRun!=null')
        record=page.evaluate('async()=>browserBackend.records(state.selectedRun)')
        assert record['config']['task']['foot_drift_weight']==1
        assert record['config']['task']['foot_only_support'] is True
        assert record['config']['task']['mechanical_work_weight']==.2
        assert record['policy']['obsSize']==57
        assert record['model_info']['action_schema']=='rlsim-joint-targets-v1'
        outcome=page.evaluate('async()=>{const r=await browserBackend.records(state.selectedRun);const before=JSON.stringify([r.policy,r.normalizer]);await browserBackend.call("validate",r);const s=await browserBackend.request("/api/replay",{run_id:r.id,seed:111});const after=await browserBackend.records(r.id);return {summary:s.summary,frozen:before===JSON.stringify([after.policy,after.normalizer])};}')
        assert len(outcome['summary']['gait']['steps_per_leg'])==4
        assert outcome['frozen']
        assert outcome['summary']['effort']['absolute_motor_work_j']>0
        continued=page.evaluate('async body=>browserBackend.request("/api/replay",{body,task:{preset:"walk",episode_seconds:1,min_height_ratio:1,terminate_on_fall:false,foot_only_support:true},target_speed:1,seed:13})',record['config']['body'])
        assert continued['summary']['steps']==50
        assert any(frame['fallen'] for frame in continued['frames']) and continued['summary']['fell']

        page.evaluate('task=>setTask(task)',record['config']['task'])
        assert page.evaluate('readTask().mechanical_work_weight')==.2
        assert page.evaluate('readTask().foot_only_support') is True
        page.evaluate('async id=>browserBackend.request("/api/resume",{run_id:id,additional_steps:4096})',record['id'])
        page.wait_for_function("state.status?.steps===8192&&state.status?.status==='completed'",timeout=180000)
        child=page.evaluate('async()=>browserBackend.records(state.status.run_id)')
        assert child['config']['task']==record['config']['task']
        assert child['continuation']['optimizer']['iterations']>record['continuation']['optimizer']['iterations']
        assert child['normalizer']['count']>record['normalizer']['count']
        assert page.evaluate('async id=>browserBackend.records(id)',record['id'])==record
        assert not errors,errors
        context.close()
        if not cdp:browser.close()


def test_coordinated_demo_keeps_weights_across_shapes(static_url):
    pw=pytest.importorskip('playwright.sync_api')
    with pw.sync_playwright() as p:
        cdp=os.getenv('MORPHOLAB_GPU_CDP')
        browser=p.chromium.connect_over_cdp(cdp,timeout=10000) if cdp else p.chromium.launch(headless=True)
        context=browser.new_context(viewport={'width':1440,'height':1000})
        if not cdp:
            def cpu(route):
                response=route.fetch();route.fulfill(response=response,body="Object.defineProperty(navigator,'gpu',{value:undefined});\n"+response.text())
            context.route('**/browser-worker.js',cpu)
        page=context.new_page();page.goto(os.getenv('MORPHOLAB_TEST_URL',static_url))
        page.wait_for_function("state.connected&&!state.initializing&&!state.previewPending&&document.querySelector('#generalization-demo').getAttribute('aria-busy')==='false'",timeout=90000)
        if cdp:assert page.evaluate('state.hardware.training_device')=='webgpu'
        page.locator('[data-tab=explore]').click()
        page.locator('#generalization-policy').select_option('coordinated')
        page.wait_for_function('!proportionBusy',timeout=90000)
        page.locator('#generalization-run').click()
        page.wait_for_function('!proportionBusy&&proportionLastResult!=null',timeout=90000)
        first=page.evaluate('proportionLastResult')
        before=page.evaluate("async()=>{const r=await browserBackend.records('generalization-coordinated');return JSON.stringify([r.policy,r.normalizer]);}")
        page.locator('[data-proportions="1,1.16,.84"]').click()
        page.wait_for_function('!proportionBusy&&proportionLastResult?.proportions[1]===1.16',timeout=90000)
        result=page.evaluate('proportionLastResult')
        assert result['body']['upper_length']==pytest.approx(.42*1.16)
        assert result['model_info']['action_schema']=='rlsim-joint-targets-v1'
        assert min(first['summary']['gait']['steps_per_leg'])>=6
        assert min(result['summary']['gait']['steps_per_leg'])>=6
        assert not first['summary']['fell'] and not result['summary']['fell']
        assert page.evaluate("async()=>{const r=await browserBackend.records('generalization-coordinated');return JSON.stringify([r.policy,r.normalizer]);}")==before
        out=Path(__file__).resolve().parents[1]/'test-results/walking'
        out.mkdir(parents=True,exist_ok=True)
        for w,h in [(1440,1000),(390,844)]:
            page.set_viewport_size({'width':w,'height':h})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            page.locator('.control-content').evaluate('e=>e.scrollTop=0')
            page.screenshot(path=str(out/f'explore-{w}.png'))
        context.close()
        if not cdp:browser.close()
