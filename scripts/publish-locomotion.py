"""Publish one fully audited broad-coverage policy without replacing older studies."""
import argparse,hashlib,json,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from morpholab.training import atomic_json,read_json
from morpholab.motion import walking_pass

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True);p.add_argument('--record',type=Path,required=True)
p.add_argument('--baseline-final',type=Path,required=True);p.add_argument('--native-final',type=Path,required=True)
p.add_argument('--cpu-final',type=Path,required=True);p.add_argument('--gpu-final',type=Path,required=True)
p.add_argument('--condition-audits',type=Path,nargs=2,required=True)
a=p.parse_args();state=read_json(a.source/'status.json');assert state['status']=='completed'
record=read_json(a.record);assert record['steps']==state['best_checkpoint_steps']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
saved_config=read_json(a.source/'config.json')
assert record['config']['device']=='auto'  # Portable files select the visitor's device.
assert {k:v for k,v in record['config'].items() if k!='device'}=={k:v for k,v in saved_config.items() if k!='device'}
native=read_json(a.native_final);baseline=read_json(a.baseline_final)
assert native['checkpoint_sha256']==sha(a.source/'best_policy.zip')
assert native['suite']==baseline['suite']=='final' and native['completed'] and baseline['completed']
assert native['seeds']==baseline['seeds']
assert native['checkpoint_steps']==record['steps']
for key in ['seeds','cases']:assert key in native
browsers=[read_json(p) for p in [a.cpu_final,a.gpu_final]]
for b in browsers:
 assert b['completed_at'] and b['weights_unchanged'] and b['suite']=='final'
 assert b['record_sha256']==sha(a.record) and b['native_audit_sha256']==sha(a.native_final)
 assert [(c['scales'],c['speed']) for c in b['cases']]==[(c['scales'],c['speed']) for c in native['cases']]
 assert b['seeds']==native['seeds']
conditions=[read_json(p) for p in a.condition_audits]
assert all(c['completed'] and c['suite']=='validation' for c in conditions)
assert not set(native['seeds']).intersection(e['seed'] for e in state['best_validation']['episodes'])
rows=[]
for label,backend,r in [('Previous coordinated','Final · native',baseline),('Broader training','Final · native',native),('Broader training','Final · browser CPU',browsers[0]),('Broader training','Final · browser GPU',browsers[1])]:
 episodes=[e for c in r['cases'] for e in c['episodes']]
 assert all(e['gait_passed']==walking_pass(e) for e in episodes)
 rows.append([label,backend,f"{sum(e['gait_passed'] for e in episodes)}/{len(episodes)}"])
root=Path(__file__).resolve().parents[1]/'morpholab/static/demos';filename='walking-broad.json'
shutil.copyfile(a.record,root/filename)
entry={'file':filename,'label':'Coordinated walking · wider range','steps':record['steps'],'training_steps':state['steps'],
 'seed':record['config']['seed'],'run_id':str(a.source),'sha256':sha(root/filename),
 'fingerprint':hashlib.sha256(json.dumps({'weights':record['policy']['weights'],'normalizer':record['normalizer']},sort_keys=True,separators=(',',':')).encode()).hexdigest(),
 'morphology':record['config']['morphology'],'selection':'Mean return on eight development cases; penalty condition chosen using the separate validation audit.',
 'validation':state['best_validation'],'training_seconds':state['elapsed']}
boundary=read_json(a.source.parent/(a.source.name+'-corners.json'))
assert boundary['completed']
be=[e for c in boundary['cases'] for e in c['episodes']]
boundary_note=f"Boundary/speed stress audit: {sum(e['gait_passed'] for e in be)}/{len(be)} passed. Extreme upper/lower-leg ratios and high-speed withheld shapes can still fail or underuse a leg."
report={'format':'rlsim-locomotion-study-v1','completed_at':datetime.now(timezone.utc).isoformat(),
 'controller':entry,'training_protocol':read_json(a.source.parent/'protocol.json'),'selection':read_json(a.source.parent/'selection.json'),'comparison_rows':rows,
 'efficiency_comparison':read_json(a.source.parent/'condition-comparison.json'),'final_comparison':read_json(a.source.parent/'final-comparison.json'),
 'post_training_notes':read_json(a.source.parent/'post-training-notes.json'),
 'condition_validation':conditions,'boundary_audit':boundary,'boundary_note':boundary_note,'baseline_final':baseline,'native_final':native,'browser_final':browsers,
 'limits':'One warm-started training seed, nominal test physics, and planar quadrupeds with eight unchanged joints. Failed cases are retained. More reliable walking does not imply minimum motor work. Final shapes and episode seeds were reserved until the condition was selected; individual episodes are not independent training runs.'}
atomic_json(root/'locomotion-study.json',report)
gallery=read_json(root/'gallery.json');assert (root/'walking-broad.png').exists()
nominal=next(c for c in browsers[0]['cases'] if c['scales']==[1,1,1] and c['speed']==1)['episodes']
card={'id':'coordinated-broad','file':filename,'title':'Quadruped · wider proportions and speeds',
 'description':'One frozen controller across body lengths and speed commands. Every leg is checked for repeated steps; knee or torso support counts as a fall. Try the sliders above.',
 'steps':entry['steps'],'seed':950200007,'provenance':f"Native PPO; {entry['training_steps']} total training steps; browser inference",
 'source_run':entry['run_id'],'checkpoint':'best','sha256':entry['sha256'],'thumbnail':'walking-broad.png',
 'browser_evaluation':{'device':'CPU','seeds':browsers[0]['seeds'],'episodes':nominal,
 'survived':sum(not e['fell'] for e in nominal),'mean_distance':sum(e['distance'] for e in nominal)/len(nominal),
 'mean_speed_error':sum(e['speed_error'] for e in nominal)/len(nominal),'gait_passed':sum(e['gait_passed'] for e in nominal)}}
gallery['entries']=[card,*[e for e in gallery['entries'] if e['id']!=card['id']]]
atomic_json(root/'gallery.json',gallery)

print(json.dumps({'controller':entry['label'],'steps':entry['steps'],'rows':rows},indent=2))
