"""Import real SkinTokens results and run the browser's physics checks.

This performs no locomotion training. Run with RLsim's development environment.
"""
import argparse
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def audit(folder):
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(QuietHandler,directory=str(ROOT/'morpholab/static')))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    results=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True)
            context=browser.new_context(viewport={'width':1440,'height':1000})
            def cpu(route):
                response=route.fetch();route.fulfill(response=response,body="Object.defineProperty(navigator,'gpu',{value:undefined});\n"+response.text())
            context.route('**/browser-worker.js',cpu)
            page=context.new_page();errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}/')
            page.wait_for_function('state.connected&&!state.initializing&&!state.previewPending',timeout=90000)
            for name in ['giraffe','humanoid','hexapod']:
                path=folder/f'{name}-rigged.glb'
                if not path.exists():
                    results.append({'body_type':name,'status':'missing-result'});continue
                try:
                    page.locator('#asset-files').set_input_files(str(path))
                    page.wait_for_function('!state.busy&&document.querySelector("#import-entry").value.endsWith("-rigged.glb")')
                    page.locator('#import-mass').fill('60' if name=='humanoid' else '10')
                    page.locator('#import-rig').select_option('skeleton')
                    page.locator('#apply-body-import').click()
                    page.wait_for_function('!state.busy&&!rigWorkflow.checking&&rigWorkflow.report&&rigWorkflow.body===state.body',timeout=90000)
                    report=page.evaluate('structuredClone(rigWorkflow.report)')
                    body=page.evaluate('structuredClone(state.body)')
                    assert body['visual']['mode']=='skinned'
                    (folder/f'{name}-body.json').write_text(json.dumps({'format':'rlsim-body-v1','body':body,'environment':{'gravity':9.81,'friction':1}}))
                    (folder/f'{name}-physics.json').write_text(json.dumps(report,indent=2))
                    page.locator('#show-colliders').check()
                    page.locator('#rig-check-details').evaluate('e=>e.open=false')
                    for width,height in [(1440,1000),(390,844)]:
                        page.set_viewport_size({'width':width,'height':height})
                        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
                        page.locator('.control-content').evaluate('e=>e.scrollTop=0')
                        page.screenshot(path=str(folder/f'{name}-{width}.png'))
                    page.set_viewport_size({'width':1440,'height':1000})
                    results.append({'body_type':name,'status':report['status'],'physical_parts':len(body['physical_rig']['bodies']),
                                    'profile':body['physical_rig']['profile'],'controls':page.evaluate('state.modelInfo.action_size'),
                                    'output_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'checks':report})
                except Exception as exc:
                    results.append({'body_type':name,'status':'failed','error':str(exc),'ui_status':page.locator('#status-text').inner_text()})
            assert not errors,errors
            context.close();browser.close()
    finally:
        server.shutdown();server.server_close()
    (folder/'browser-audit.json').write_text(json.dumps({'format':'rlsim-auto-rig-audit-v1','device':'browser CPU physics','locomotion_trained':False,'results':results},indent=2))
    print(json.dumps([{k:v for k,v in result.items() if k!='checks'} for result in results],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args();audit(args.folder.resolve())
