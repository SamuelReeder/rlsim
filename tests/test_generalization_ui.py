"""Real portable-policy inference: edit proportions without updating weights."""
import json
import os
from pathlib import Path
import pytest
from test_browser_local import static_url

pytestmark=pytest.mark.skipif(not os.getenv('MORPHOLAB_BROWSER_TEST'),reason='Opt-in generalization browser checks')


def test_proportion_override_uses_new_physics_and_keeps_checkpoint(static_url):
    pw=pytest.importorskip('playwright.sync_api')
    with pw.sync_playwright() as runtime:
        browser=runtime.chromium.launch(headless=True)
        context=browser.new_context()
        def cpu(route):
            response=route.fetch();route.fulfill(response=response,body="Object.defineProperty(navigator,'gpu',{value:undefined});\n"+response.text())
        context.route('**/browser-worker.js',cpu)
        page=context.new_page();page.goto(static_url)
        page.wait_for_function('state.connected&&!state.initializing&&!state.previewPending',timeout=90000)
        result=page.evaluate("""async()=>{
          const r=await (await fetch('./demos/quadruped-walk.json')).json();
          r.config.morphology={enabled:false,width:.2,ramp_steps:0};r.id='test-proportions-fixed';
          await browserBackend.put(r);
          const before=JSON.stringify([r.policy,r.normalizer]);
          const replay=proportions=>browserBackend.request('/api/replay',{run_id:r.id,proportions,seed:17});
          const nominal=await replay([1,1,1]);
          const changed=await replay([.85,1.18,.82]);
          const repeated=await replay([.85,1.18,.82]);
          const saved=await browserBackend.records(r.id);
          return {nominal:nominal.body,changed:changed.body,
            geometryChanged:JSON.stringify(nominal.frames[0].geoms.map(g=>g.size))!==JSON.stringify(changed.frames[0].geoms.map(g=>g.size)),
            reused:repeated.inference_cache_hit,changedCache:changed.inference_cache_hit,
            same:before===JSON.stringify([saved.policy,saved.normalizer]),
            deterministic:JSON.stringify(changed.summary)===JSON.stringify(repeated.summary)};
        }""")
        assert result['geometryChanged'] and result['same'] and result['deterministic']
        assert result['reused'] and not result['changedCache']
        assert result['nominal']['torso_length']==.8
        assert result['changed']['torso_length']==pytest.approx(.8*.85)
        context.close();browser.close()


def test_frozen_policy_across_proportions_and_baseline(static_url):
    pw=pytest.importorskip('playwright.sync_api')
    with pw.sync_playwright() as runtime:
        cdp=os.getenv('MORPHOLAB_GPU_CDP')
        browser=runtime.chromium.connect_over_cdp(cdp) if cdp else runtime.chromium.launch(headless=True)
        context=browser.new_context(viewport={'width':1440,'height':1000})
        if not cdp:
            def cpu(route):
                response=route.fetch();route.fulfill(response=response,body="Object.defineProperty(navigator,'gpu',{value:undefined});\n"+response.text())
            context.route('**/browser-worker.js',cpu)
        page=context.new_page();errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(os.getenv('MORPHOLAB_TEST_URL',static_url))
        page.wait_for_function('state.connected&&!state.previewPending&&!state.initializing&&proportionDemo!==null',timeout=90000)
        if cdp:assert page.evaluate('state.hardware.training_device')=='webgpu'
        # Any attempt to train in this demo fails the test.
        page.evaluate("""() => {
          globalThis.proportionOperations=[];
          const call=browserBackend.call.bind(browserBackend);
          browserBackend.call=(op,...args)=>{proportionOperations.push(op);if(['train','resume'].includes(op))throw new Error('Demo attempted training');return call(op,...args);};
        }""")
        page.locator('[data-tab=explore]').click()
        page.wait_for_function("document.querySelector('#generalization-demo').getAttribute('aria-busy')==='false'")
        page.locator('#generalization-auto').uncheck()
        page.locator('#generalization-policy').select_option('varied')
        page.wait_for_function('!proportionBusy')
        page.locator('#generalization-auto').check()
        page.locator('#generalization-run').click()
        page.wait_for_function('!proportionBusy&&proportionLastResult!=null',timeout=90000)
        first=page.evaluate('proportionLastResult')
        assert first['body']['legs']==4
        original=page.evaluate("async()=>{const r=await browserBackend.records('generalization-varied');return JSON.stringify([r.policy,r.normalizer]);}")
        identity=page.locator('#generalization-identity').inner_text()
        page.locator('[data-proportions="1,1.16,.84"]').click()
        page.wait_for_function('!proportionBusy&&proportionLastResult?.proportions[1]===1.16',timeout=90000)
        withheld=page.evaluate('proportionLastResult')
        assert withheld['body']['upper_length']==pytest.approx(.42*1.16)
        assert withheld['body']['lower_length']==pytest.approx(.42*.84)
        assert withheld['model_info']['action_size']==8
        assert 'Withheld region' in page.locator('#proportion-region').inner_text()
        assert page.locator('#generalization-identity').inner_text()==identity
        assert page.evaluate("async()=>{const r=await browserBackend.records('generalization-varied');return JSON.stringify([r.policy,r.normalizer]);}")==original
        # A slider release really applies a new shape, rather than reusing the cached body.
        page.locator('#proportion-torso').evaluate("e=>{e.value='1.3';e.dispatchEvent(new Event('input',{bubbles:true}));}")
        page.locator('#proportion-torso').dispatch_event('change')
        page.wait_for_function('!proportionBusy&&proportionLastResult?.proportions[0]===1.3',timeout=90000)
        assert page.evaluate('proportionLastResult.body.torso_length')==pytest.approx(.8*1.3)
        assert 'Outside' in page.locator('#proportion-region').inner_text()
        page.locator('#generalization-policy').select_option('fixed')
        page.wait_for_function("!proportionBusy&&proportionLastResult?.controller==='fixed'",timeout=90000)
        assert page.evaluate('proportionLastResult.proportions')==[1.3,1.16,.84]
        assert page.evaluate('proportionLastResult.seed')==first['seed']
        page.locator('#generalization-evidence > summary').click()
        assert page.locator('#generalization-evidence-content tbody tr').count() in [12,18,22]
        output=Path(__file__).resolve().parents[1]/'test-results/generalization';output.mkdir(parents=True,exist_ok=True)
        for w,h in [(1440,1000),(390,844)]:
            page.set_viewport_size({'width':w,'height':h})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            page.locator('#generalization-demo').scroll_into_view_if_needed()
            page.screenshot(path=str(output/f'demo-{w}.png'))
        assert not errors,errors
        assert not set(page.evaluate('proportionOperations')).intersection(['train','resume'])
        context.close()
        if not cdp:browser.close()
