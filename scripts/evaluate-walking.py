"""Export validation-selected walking controllers and retain all browser checks."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright
from morpholab.training import atomic_json,read_json,evaluate_policy,load_checkpoint
from morpholab.portable import export_checkpoint
from morpholab.morphology import body_at,region
from morpholab.motion import walking_pass
from morphology_study import TEST_BODIES

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--study',type=Path,default=Path('experiments/gait-v1'))
p.add_argument('--refinement',type=Path)
p.add_argument('--url',default='http://localhost:8777/')
args=p.parse_args()
root=Path(__file__).resolve().parents[1]/'morpholab/static/demos'
seeds=[930000007,930000021,930000039]
report={'format':'rlsim-walking-study-v1','protocol':read_json(args.study/'protocol.json'),
        'evaluation_seeds':seeds,'controllers':{},'cases':[],
        'gait_rule':'20 s without falling; mean speed error <=0.35 m/s; >=6 completed steps per leg; 15–85% grounded time per leg; foot-center drift <=0.35 m/s; pitch RMS <=0.25 rad.',
        'limits':'One new training seed per condition. Multiple training changes together. Reuses the previous proportion benchmark with fresh episode seeds. Foot-center drift includes spherical-foot rolling. Original controllers are historical baselines, not a matched ablation.'}
for name in ['quadruped','varied']:
    folder=args.refinement if name=='varied' and args.refinement else args.study/name;state=read_json(folder/'status.json')
    if state['status']!='completed':raise SystemExit(f'{name} is not completed')
    if (folder/'fine_tuning.json').exists():report['refinement']=read_json(folder/'fine_tuning.json')
    filename=f'walking-{name}.json';record=export_checkpoint(folder,root/filename,checkpoint='best')
    fingerprint=hashlib.sha256(json.dumps({'weights':record['policy']['weights'],'normalizer':record['normalizer']},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    report['controllers'][name]={'file':filename,'label':'Coordinated walking · '+name,'steps':record['steps'],
        'training_steps':state['steps'],'seed':record['config']['seed'],'run_id':str(folder),'sha256':hashlib.sha256((root/filename).read_bytes()).hexdigest(),
        'fingerprint':fingerprint,'selection':'Highest mean return on separate validation episodes','validation':state['best_validation'],'training_seconds':state['elapsed']}
    config,model,norm=load_checkpoint(folder,'best')
    try:
        for label,scales in TEST_BODIES if name=='varied' else TEST_BODIES[:1]:
            c=config.model_copy(update={'body':body_at(config.body,scales)})
            episodes=[evaluate_policy(c,model,norm,seed=seed)['summary'] for seed in seeds]
            report['cases'].append({'backend':'native','controller':name,'name':label,'scales':scales,'region':region(scales),'episodes':episodes,'gait_passed':sum(map(walking_pass,episodes))})
    finally:norm.close()
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True);context=browser.new_context(viewport={'width':1440,'height':1000})
    def cpu(route):
        response=route.fetch();route.fulfill(response=response,body="Object.defineProperty(navigator,'gpu',{value:undefined});\n"+response.text())
    context.route('**/browser-worker.js',cpu)
    page=context.new_page();page.goto(args.url)
    page.wait_for_function('state.connected&&!state.previewPending&&!state.initializing',timeout=90000)
    for name,file,varied in [('quadruped','walking-quadruped.json',False),('varied','walking-varied.json',True),('old quadruped','quadruped-walk.json',False),('old varied','proportions-varied.json',True)]:
        if name.startswith('old '):report.setdefault('baselines',{})[name]={'file':file,'sha256':hashlib.sha256((root/file).read_bytes()).hexdigest()}
        record=read_json(root/file);record['id']='walking-check-'+name
        page.evaluate('async record=>{await browserBackend.call("validate",record);await browserBackend.put(record);}',record)
        before=page.evaluate('async id=>{const r=await browserBackend.records(id);return JSON.stringify([r.policy,r.normalizer]);}',record['id'])
        for label,scales in TEST_BODIES if varied else TEST_BODIES[:1]:
            episodes=[]
            for seed in seeds:
                payload={'run_id':record['id'],'seed':seed,'checkpoint':'latest'}
                if varied:payload['proportions']=scales
                episodes.append(page.evaluate('async p=>(await browserBackend.request("/api/replay",p)).summary',payload))
            report['cases'].append({'backend':'browser CPU','controller':name,'name':label,'scales':scales,'region':region(scales),'episodes':episodes,'gait_passed':sum(map(walking_pass,episodes))})
            print(name,label,'gait',sum(map(walking_pass,episodes)),'steps',[e['gait']['steps_per_leg'] for e in episodes],flush=True)
        after=page.evaluate('async id=>{const r=await browserBackend.records(id);return JSON.stringify([r.policy,r.normalizer]);}',record['id'])
        assert before==after,'Inference modified policy weights or normalization.'
    context.close();browser.close()
report['completed_at']=datetime.now(timezone.utc).isoformat()
atomic_json(root/'walking-study.json',report)
atomic_json(args.study/'evaluation.json',report)
