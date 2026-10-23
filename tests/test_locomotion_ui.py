"""Exercise the published frozen policy through the actual proportion and speed controls."""
import json,os
from pathlib import Path
import pytest
from test_browser_local import static_url
pytestmark=pytest.mark.skipif(not os.getenv('MORPHOLAB_BROWSER_TEST'),reason='Opt-in full browser inference')


def test_broad_policy_shape_speed_and_replay_are_frozen(static_url):
    pw=pytest.importorskip('playwright.sync_api')
    with pw.sync_playwright() as p:
        cdp=os.getenv('MORPHOLAB_GPU_CDP')
        b=p.chromium.connect_over_cdp(cdp) if cdp else p.chromium.launch(headless=True)
        c=b.new_context(viewport={'width':1440,'height':1000})
        if not cdp:
            def cpu(route):
                response=route.fetch();route.fulfill(response=response,body="Object.defineProperty(navigator,'gpu',{value:undefined});\n"+response.text())
            c.route('**/browser-worker.js',cpu)
        page=c.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(os.getenv('MORPHOLAB_TEST_URL',static_url));page.wait_for_function('state.connected&&!state.initializing&&!state.previewPending&&proportionDemo?.controllers.robust&&document.querySelector("#generalization-demo").getAttribute("aria-busy")==="false"',timeout=90000)
        page.locator('[data-tab=explore]').click();assert page.locator('#generalization-policy').input_value()=='robust'
        assert page.locator('#generalization-speed-field').is_visible()
        assert '60–140%' in page.locator('#proportion-region').inner_text()
        page.evaluate("""()=>{const call=browserBackend.call.bind(browserBackend);browserBackend.call=(op,...args)=>{if(['train','resume'].includes(op))throw new Error('Inference attempted training');return call(op,...args);};}""")
        page.locator('#generalization-run').click();page.wait_for_function('!proportionBusy&&proportionLastResult',timeout=90000)
        snapshot=page.evaluate('async()=>{const r=await browserBackend.records("generalization-robust");return JSON.stringify([r.policy,r.normalizer]);}')
        page.locator('#generalization-speed').evaluate("e=>{e.value='.75';e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));}")
        page.wait_for_function('!proportionBusy&&proportionLastResult?.target_speed===.75',timeout=90000)
        page.locator('[data-proportions="1,1.16,.84"]').click();page.wait_for_function('!proportionBusy&&proportionLastResult?.proportions[1]===1.16',timeout=90000)
        result=page.evaluate('proportionLastResult')
        assert result['inference_device']==('webgpu' if cdp else 'cpu')
        assert result['target_speed']==.75 and result['body']['upper_length']==pytest.approx(.42*1.16)
        assert not result['summary']['fell'] and min(result['summary']['gait']['steps_per_leg'])>=6
        assert 'Withheld region' in page.locator('#proportion-region').inner_text()
        assert result['summary']['effort']['absolute_motor_work_j']>0
        assert snapshot==page.evaluate('async()=>{const r=await browserBackend.records("generalization-robust");return JSON.stringify([r.policy,r.normalizer]);}')
        out=Path(__file__).resolve().parents[1]/'test-results/locomotion';out.mkdir(parents=True,exist_ok=True)
        for width,height in [(1440,1000),(390,844)]:
            page.set_viewport_size({'width':width,'height':height});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            page.locator('.control-content').evaluate('e=>e.scrollTop=0');page.screenshot(path=str(out/f'broad-explore-{width}.png'))
        page.locator('#generalization-policy').select_option('coordinated');page.wait_for_function('!proportionBusy&&proportionLastResult?.controller==="coordinated"',timeout=90000)
        assert page.locator('#generalization-speed-field').is_hidden();assert 'Withheld region' in page.locator('#proportion-region').inner_text()
        assert page.evaluate('proportionLastResult.target_speed')==1
        assert not errors,errors
        c.close()
        if not cdp:b.close()
