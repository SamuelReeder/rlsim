"""Capture the actual trained-policy renderer for gallery thumbnails and layout QA."""
from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[1];out=root/'test-results/walking';out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
 b=p.chromium.launch(headless=True);c=b.new_context(viewport={'width':1440,'height':1000})
 def cpu(route):
  response=route.fetch();route.fulfill(response=response,body="Object.defineProperty(navigator,'gpu',{value:undefined});\n"+response.text())
 c.route('**/browser-worker.js',cpu);page=c.new_page();page.goto('http://localhost:8777/')
 page.wait_for_function('state.connected&&!state.initializing&&!state.previewPending',timeout=90000)
 for name in ['quadruped','varied']:
  result=page.evaluate('''async name=>{const r=await (await fetch('./demos/walking-'+name+'.json')).json();r.id='capture-'+name;await browserBackend.call('validate',r);await browserBackend.put(r);await refreshRuns(r.id);$('eval-seed').value=940000007;$('eval-speed').value=1;selectTab('run',false);return await infer();}''',name)
  page.evaluate('state.playing=false;seek(3.32)');page.wait_for_timeout(200)
  page.locator('#simulation').screenshot(path=str(root/'morpholab/static/demos'/f'walking-{name}.png'))
  page.screenshot(path=str(out/f'{name}-desktop.png'))
  print(name,result['summary'],flush=True)
 c.close();b.close()
