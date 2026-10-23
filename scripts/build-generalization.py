"""Bundle validation-selected controllers and every final test result."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from morpholab.portable import export_checkpoint
from morpholab.training import atomic_json, read_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--study',type=Path,default=Path('experiments/proportions-v1'))
args=p.parse_args()
report=read_json(args.study/'report.json')
if not report.get('completed_at'):
    raise SystemExit('Finish all planned training and evaluation before publishing.')
output=Path(__file__).resolve().parents[1]/'morpholab/static/demos'
manifest={'format':'rlsim-proportion-demo-v1','width':report['protocol']['width'],
          'demo_seed':920000007,'controllers':{},'notes':[
              'The demo checkpoints were selected by validation return, not final test scores. The full report includes all training seeds.',
              'Both conditions used the same dynamics randomization. Only the varied condition changed body proportions during training.',
              'Three training seeds are a small experiment. Success here does not establish transfer to other skeletons, arbitrary meshes or physical robots.']}
for key,condition in [('varied','varied bodies'),('fixed','fixed body')]:
    candidates=[t for t in report['trials'] if t['condition']==condition and t['status']=='completed']
    if not candidates:raise SystemExit(f'No completed trials for {condition}; cannot publish a demo.')
    selected=max(candidates,key=lambda t:t['validation']['mean_return'])
    folder=args.study/'runs'/selected['run_id']
    filename=f'proportions-{key}.json'
    record=export_checkpoint(folder,output/filename,checkpoint='best')
    fingerprint=hashlib.sha256(json.dumps({'weights':record['policy']['weights'],'normalizer':record['normalizer']},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    manifest['base_body']=record['config']['body']
    manifest['controllers'][key]={'file':filename,'label':f'PPO · {condition}',
        'steps':record['steps'],'training_steps':selected['steps'],'seed':selected['seed'],'run_id':selected['run_id'],
        'sha256':hashlib.sha256((output/filename).read_bytes()).hexdigest(),'fingerprint':fingerprint,
        'selection':'Highest validation return; held-out test scores not used.'}
atomic_json(output/'generalization-report.json',report)
atomic_json(output/'generalization.json',manifest)
print(json.dumps(manifest['controllers'],indent=2))
