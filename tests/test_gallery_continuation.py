"""Gallery, round-trip training continuation and saved-model comparisons in a real browser."""
import os
from pathlib import Path
import pytest
from test_browser_local import static_url

pytestmark=pytest.mark.skipif(not os.getenv('MORPHOLAB_BROWSER_TEST'),reason='Opt-in gallery and continuation')

def test_gallery_and_training_continuation_after_reload(static_url):
    pw=pytest.importorskip('playwright.sync_api')
    with pw.sync_playwright() as runtime:
        cdp=os.getenv('MORPHOLAB_GPU_CDP')
        browser=runtime.chromium.connect_over_cdp(cdp) if cdp else runtime.chromium.launch(headless=True)
        context=browser.new_context(viewport={'width':1280,'height':900})
        def cpu(route):
            response=route.fetch();route.fulfill(response=response,body="Object.defineProperty(navigator,'gpu',{value:undefined});\n"+response.text())
        if not cdp:context.route('**/browser-worker.js',cpu)
        page=context.new_page();errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(os.getenv('MORPHOLAB_TEST_URL',static_url))
        page.wait_for_function('state.connected && !state.initializing && !state.previewPending',timeout=90000)
        if cdp:assert page.evaluate('state.hardware.training_device')=='webgpu'
        page.locator('[data-tab=explore]').click()
        page.wait_for_selector('[data-demo="quadruped-walk"]')
        page.locator('[data-demo="quadruped-walk"]').click()
        page.wait_for_function("!state.busy && state.frames.length>2",timeout=90000)
        assert page.evaluate("state.selectedRun")=='gallery-quadruped-walk'
        assert page.evaluate('state.frames.at(-1).position-state.frames[0].position')>10
        assert page.locator('#resume-training').is_disabled()
        gallery=page.evaluate("browserBackend.records('gallery-quadruped-walk')")
        assert 'continuation' not in gallery
        page.evaluate("""() => {
          selectTab('train',false);setTask({preset:'velocity',episode_seconds:1,terminate_on_fall:false});
          setAdvanced({hidden_size:32,epochs:1,best_checkpoint:false,curriculum_mode:'staged'});
          $('training-budget').value=4096;$('n-envs').value=2;return train();
        }""")
        page.wait_for_function("['completed','failed'].includes(state.status.status)",timeout=120000)
        assert page.evaluate('state.status.status')=='completed',page.locator('#status-text').inner_text()
        parent=page.evaluate('state.status.run_id')
        original=page.evaluate('id=>browserBackend.records(id)',parent)
        assert original['continuation']['optimizer']['iterations']>0
        assert original['continuation']['environment_random']
        page.reload()
        page.wait_for_function('state.connected && !state.initializing && !state.previewPending',timeout=90000)
        page.locator('[data-tab=run]').click()
        page.locator('#saved-run').select_option(parent)
        page.locator('#continue-training > summary').click()
        page.locator('#additional-steps').fill('4096')
        page.wait_for_function('!state.previewPending')
        page.locator('#resume-training').click()
        page.wait_for_function("['completed','failed'].includes(state.status.status) && state.status.run_id!=='"+parent+"'",timeout=120000)
        assert page.evaluate('state.status.status')=='completed',page.locator('#status-text').inner_text()
        child=page.evaluate('id=>browserBackend.records(id)',page.evaluate('state.status.run_id'))
        assert child['steps']==8192 and child['start_steps']==4096 and child['parent_run_id']==parent
        assert child['continuation']['optimizer']['iterations']>original['continuation']['optimizer']['iterations']
        assert child['normalizer']['count']>original['normalizer']['count']
        assert child['continuation']['schedule_steps']==4096
        assert child['continuation']['curriculum']['index']>=original['continuation']['curriculum']['index']
        assert page.evaluate('id=>browserBackend.records(id)',parent)==original
        # Check import/export retains the optimizer and validates its compatibility.
        imported=page.evaluate("""async id=>{
          const record=await browserBackend.records(id);
          const checked=await browserBackend.call('validate',JSON.parse(JSON.stringify(record)));
          return checked.continuation;
        }""",child['id'])
        assert imported==child['continuation']
        page.locator('[data-tab=explore]').click()
        page.locator('#compare-models > summary').click()
        for id in [parent,child['id']]:page.locator(f'[data-compare-model][value="{id}"]').check()
        page.locator('#run-comparison').click()
        page.wait_for_function("!state.busy && comparisonData?.status==='completed'",timeout=90000)
        report=page.evaluate('comparisonData')
        assert len(report['runs'])==2 and all(len(r['episodes'])==3 for r in report['runs'])
        assert report['warnings'] and all(r['status']=='completed' for r in report['runs'])
        # Cancelling a comparison must also cancel its active replay, rather than
        # merely skipping later models. Use the longer gallery episode here.
        checked=page.locator('[data-compare-model]:checked').evaluate_all('inputs=>inputs.map(input=>input.value)')
        for id in checked:page.locator(f'[data-compare-model][value="{id}"]').uncheck()
        for id in ['gallery-quadruped-walk',child['id']]:page.locator(f'[data-compare-model][value="{id}"]').check()
        page.locator('#run-comparison').click()
        page.locator('#stop-comparison').click()
        page.wait_for_function('!state.busy',timeout=30000)
        assert page.evaluate('comparisonData.status')=='stopped'
        assert page.evaluate("comparisonData.runs.some(run=>run.status==='stopped')")
        assert page.evaluate('browserBackend.pending.size')<=1  # Status polling may be in flight.
        output=Path(__file__).resolve().parents[1]/'test-results'/'gallery';output.mkdir(parents=True,exist_ok=True)
        for width,height in [(1440,1000),(390,844)]:
            page.set_viewport_size({'width':width,'height':height})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            page.screenshot(path=str(output/f'explore-{width}.png'),full_page=True)
        assert not errors,errors
        context.close();browser.close()
