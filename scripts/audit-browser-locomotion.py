"""Replay a frozen portable policy on an audit's shapes, speeds and seeds."""
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright
from morpholab.motion import walking_pass
from morpholab.training import atomic_json,read_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--record',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--url',default='http://localhost:8777/')
p.add_argument('--cdp');p.add_argument('--max-cases',type=int)
a=p.parse_args();record=read_json(a.record);audit=read_json(a.audit)
report={'format':'rlsim-browser-locomotion-audit-v1','record_sha256':hashlib.sha256(a.record.read_bytes()).hexdigest(),
 'backend':'browser WebGPU' if a.cdp else 'browser CPU','native_audit_sha256':hashlib.sha256(a.audit.read_bytes()).hexdigest(),
 'suite':audit['suite'],'seeds':audit['seeds'],'cases':[]}
with sync_playwright() as pw:
 browser=pw.chromium.connect_over_cdp(a.cdp) if a.cdp else pw.chromium.launch(headless=True)
 context=browser.new_context(viewport={'width':1440,'height':1000})
 if not a.cdp:
  def cpu(route):
   response=route.fetch();route.fulfill(response=response,body="Object.defineProperty(navigator,'gpu',{value:undefined});\n"+response.text())
  context.route('**/browser-worker.js',cpu)
 page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.goto(a.url)
 page.wait_for_function('state.connected&&!state.initializing&&!state.previewPending',timeout=90000)
 report['hardware']=page.evaluate('state.hardware')
 assert report['hardware']['training_device']==('webgpu' if a.cdp else 'cpu')
 record['id']='locomotion-audit'
 page.evaluate('async r=>{await browserBackend.call("validate",r);await browserBackend.put(r);}',record)
 frozen=page.evaluate('async()=>{const r=await browserBackend.records("locomotion-audit");return JSON.stringify([r.policy,r.normalizer]);}')
 for case in audit['cases'][:a.max_cases]:
  episodes=[]
  for seed in report['seeds']:
   payload={'run_id':record['id'],'proportions':case['scales'],'target_speed':case['speed'],'seed':seed}
   r=page.evaluate('async p=>{const r=await browserBackend.request("/api/replay",p);return {summary:r.summary,device:r.inference_device,target_speed:r.target_speed};}',payload)
   assert r['target_speed']==case['speed'],r
   episode=r['summary'];episode['gait_passed']=walking_pass(episode);episodes.append(episode)
  report['cases'].append({'scales':case['scales'],'speed':case['speed'],'episodes':episodes})
  atomic_json(a.output,report)
  print(case['scales'],case['speed'],'passed',sum(e['gait_passed'] for e in episodes),'/',len(episodes),flush=True)
 report['weights_unchanged']=frozen==page.evaluate('async()=>{const r=await browserBackend.records("locomotion-audit");return JSON.stringify([r.policy,r.normalizer]);}')
 assert report['weights_unchanged'];assert not errors,errors
 report['completed_at']=datetime.now(timezone.utc).isoformat();atomic_json(a.output,report)
 context.close()
 if not a.cdp:browser.close()
