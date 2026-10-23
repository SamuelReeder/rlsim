"""Evaluate the two frozen showcase policies in real browser physics."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright
from morpholab.training import atomic_json,read_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--url',default='http://localhost:8777/')
args=p.parse_args()
root=Path(__file__).resolve().parents[1]/'morpholab/static/demos'
manifest=read_json(root/'generalization.json');report=read_json(root/'generalization-report.json')
result={'format':'rlsim-proportion-browser-check-v1','device':'CPU','cases':[],'controllers':manifest['controllers']}
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True)
    context=browser.new_context(viewport={'width':1440,'height':1000})
    def cpu(route):
        response=route.fetch();route.fulfill(response=response,body="Object.defineProperty(navigator,'gpu',{value:undefined});\n"+response.text())
    context.route('**/browser-worker.js',cpu)
    page=context.new_page();page.goto(args.url)
    page.wait_for_function('state.connected&&!state.previewPending&&!state.initializing&&proportionDemo!==null',timeout=90000)
    for controller in ['varied','fixed']:
        page.evaluate('async key=>{const r=await generalizationRecord(key);await browserBackend.put(r);}',controller)
        before=page.evaluate("async key=>JSON.stringify((await browserBackend.records('generalization-'+key)).policy)",controller)
        for case in report['protocol']['body_cases']:
            episodes=[]
            for seed in report['protocol']['evaluation_seeds']:
                episode=page.evaluate("async p=>(await browserBackend.request('/api/replay',{run_id:'generalization-'+p.controller,proportions:p.scales,seed:p.seed,checkpoint:'latest'})).summary",{'controller':controller,'scales':case['scales'],'seed':seed})
                episodes.append(episode)
            result['cases'].append({'controller':controller,**case,'episodes':episodes,
                'success_rate':sum(not e['fell'] and e['speed_error']<=.35 and e['seconds']>=19.99 for e in episodes)/len(episodes)})
            print(controller,case['name'],result['cases'][-1]['success_rate'],flush=True)
        after=page.evaluate("async key=>JSON.stringify((await browserBackend.records('generalization-'+key)).policy)",controller)
        assert before==after,'Inference modified saved weights.'
    context.close();browser.close()
atomic_json(root/'generalization-browser.json',result)
