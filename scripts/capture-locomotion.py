"""Capture actual frozen-policy playback for the broad-coverage gallery entry."""
import argparse,json
from pathlib import Path
from playwright.sync_api import sync_playwright
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--record',type=Path,required=True);p.add_argument('--url',default='http://localhost:8777/')
a=p.parse_args();root=Path(__file__).resolve().parents[1];out=root/'test-results/locomotion';out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as pw:
 b=pw.chromium.launch(headless=True);c=b.new_context(viewport={'width':1440,'height':1000})
 def cpu(route):
  response=route.fetch();route.fulfill(response=response,body="Object.defineProperty(navigator,'gpu',{value:undefined});\n"+response.text())
 c.route('**/browser-worker.js',cpu);page=c.new_page();page.goto(a.url);page.wait_for_function('state.connected&&!state.initializing&&!state.previewPending',timeout=90000)
 record=json.loads(a.record.read_text());record['id']='locomotion-capture'
 result=page.evaluate('''async r=>{await browserBackend.call('validate',r);await browserBackend.put(r);await refreshRuns(r.id);$('eval-seed').value=950200007;$('eval-speed').value=1;selectTab('run',false);return await infer();}''',record)
 assert not result['summary']['fell'] and min(result['summary']['gait']['steps_per_leg'])>=6
 for t in [3.0,3.16,3.32,3.48]:
  page.evaluate('t=>{state.playing=false;seek(t);}',t);page.wait_for_timeout(100)
  page.locator('#simulation').screenshot(path=str(out/f'broad-gait-{t:.2f}.png'))
 page.locator('#simulation').screenshot(path=str(root/'morpholab/static/demos/walking-broad.png'))
 c.close();b.close()
 print(json.dumps(result['summary'],indent=2))
