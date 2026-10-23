"""Prepared examples and unavailable-service UX on ordinary static hosting."""
import os
import pytest
from test_browser_local import static_url

pytestmark=pytest.mark.skipif(not os.getenv('MORPHOLAB_BROWSER_TEST'),reason='Opt-in real browser rigging UI')


def test_prepared_examples_work_without_a_gpu_service(static_url):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        context=browser.new_context(viewport={'width':390,'height':844})
        def cpu(route):
            response=route.fetch();route.fulfill(response=response,body="Object.defineProperty(navigator,'gpu',{value:undefined});\n"+response.text())
        context.route('**/browser-worker.js',cpu)
        page=context.new_page();errors=[];requests=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:requests.append(r.url) if '/api/' in r.url else None)
        page.goto(static_url)
        page.wait_for_function('state.connected&&!state.initializing&&!state.previewPending',timeout=90000)
        page.locator('#optional-rigging > summary').click()
        for name,count in [('humanoid',28),('giraffe',47),('hexapod',14)]:
            page.locator('#rig-example').select_option(name)
            page.locator('#load-rig-example').click()
            page.wait_for_function('name=>!state.busy&&document.querySelector("#import-entry").value===`${name}-skintokens.glb`',arg=name)
            assert page.locator('#import-mass').input_value()==('60' if name=='humanoid' else '10')
            page.locator('#apply-body-import').click()
            page.wait_for_function('!state.busy&&rigWorkflow.report&&rigWorkflow.body===state.body',timeout=90000)
            assert page.evaluate('state.body.metadata.auto_rig.joints')==count
            assert page.evaluate('rigWorkflow.report.status')=='review'
            assert page.locator('#train-button').is_enabled()
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        assert not requests,requests
        # Environment edits invalidate the report and require a fresh check.
        page.locator('#gravity').fill('8')
        page.locator('#gravity').dispatch_event('change')
        page.wait_for_function('!state.previewPending&&!state.busy')
        assert page.locator('#train-button').is_disabled()
        page.locator('#run-rig-checks').click()
        page.wait_for_function('!rigWorkflow.checking&&rigWorkflow.report')
        assert page.locator('#train-button').is_enabled()
        page.locator('#check-rig-service').click()
        page.wait_for_function('document.querySelector("#auto-rig-status").textContent.includes("unavailable")')
        assert page.locator('#start-auto-rig').is_disabled()
        assert not errors,errors
        context.close();browser.close()
