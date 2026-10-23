"""Verify every bundled demo with real browser physics and capture honest thumbnails."""
import argparse
from pathlib import Path
import sys
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright
from morpholab.training import atomic_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--url',default='http://localhost:8777/')
args=p.parse_args();root=Path(__file__).resolve().parents[1]/'morpholab/static/demos'
manifest=json.loads((root/'gallery.json').read_text())
seeds=[500000007,500000021,500000039]
with sync_playwright() as runtime:
    browser=runtime.chromium.launch(headless=True)
    context=browser.new_context(viewport={'width':1150,'height':760})
    def cpu(route):
        response=route.fetch();route.fulfill(response=response,body="Object.defineProperty(navigator,'gpu',{value:undefined});\n"+response.text())
    context.route('**/browser-worker.js',cpu)
    page=context.new_page();page.goto(args.url)
    page.wait_for_function('state.connected&&!state.initializing&&!state.previewPending',timeout=90000)
    for entry in manifest['entries']:
        page.evaluate('async id=>watchDemo(galleryEntries.find(e=>e.id===id))',entry['id'])
        page.wait_for_function('!state.busy&&state.frames.length>1',timeout=90000)
        page.evaluate('state.playing=false;seek(3)')
        page.wait_for_timeout(250)
        thumbnail=f"{entry['id']}.png"
        page.locator('#simulation').screenshot(path=str(root/thumbnail))
        results=page.evaluate('''async seeds=>{
          const results=[];
          for(const seed of seeds){const r=await browserBackend.request('/api/replay',{run_id:state.selectedRun,seed,checkpoint:'latest'});results.push(r.summary);}
          return results;
        }''',seeds)
        survived=sum(not r['fell'] for r in results)
        entry['thumbnail']=thumbnail
        entry['browser_evaluation']={'device':'CPU','seeds':seeds,'episodes':results,'survived':survived,
            'mean_distance':sum(r['distance'] for r in results)/len(results),
            'mean_speed_error':sum(r['speed_error'] for r in results)/len(results)}
        print(json.dumps({'demo':entry['id'],'survived':survived,'distance':entry['browser_evaluation']['mean_distance']}),flush=True)
        if entry['id'].endswith('-walk') and (survived!=len(seeds) or entry['browser_evaluation']['mean_distance']<5):
            raise RuntimeError(f"{entry['id']} failed the browser speed-and-survival gate.")
    context.close();browser.close()
atomic_json(root/'gallery.json',manifest)
