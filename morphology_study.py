"""Train paired fixed-body / varied-body policies, then evaluate a frozen suite."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time
import uuid

from morpholab.training import TrainConfig, atomic_json, read_json, load_checkpoint, evaluate_policy
from morpholab.morphology import body_at, region, validation_scales
from morpholab.experiments import summarize

TEST_BODIES = [
    ("Nominal", [1,1,1]), ("Long torso", [1.17,1.02,1.04]),
    ("Short torso", [.83,.98,1.06]), ("Long legs", [.96,1.16,1.18]),
    ("Short legs", [1.05,.84,.82]), ("Mixed legs", [.92,.87,1.13]),
    ("Withheld A", [1,1.16,.84]), ("Withheld B", [.85,1.18,.82]),
    ("Withheld C", [1.15,1.12,.88]),
    ("Outside: short torso", [.65,1,1]),
    ("Outside: long legs", [1,1.35,1.3]),
    ("Outside: short legs", [1,.65,.65]),
]
TEST_SEEDS = [910000007, 910000021, 910000039]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1048576)
    parser.add_argument('--seeds', nargs='+', type=int, default=[7,17,27])
    parser.add_argument('--workers', type=int, choices=[1,2], default=2)
    args = parser.parse_args()
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    sources = {name: hashlib.sha256((Path('morpholab')/name).read_bytes()).hexdigest()
               for name in ['training.py','environment.py','morphology.py','config.py','tasks.py']}
    protocol = {'steps': args.steps, 'training_seeds': args.seeds, 'evaluation_seeds': TEST_SEEDS,
        'body_cases': [{'name':n,'scales':s,'region':region(s)} for n,s in TEST_BODIES],
        'conditions':['fixed body','varied bodies'], 'width':.2, 'ramp_steps':131072,
        'validation_scales':validation_scales(), 'checkpoint':'best', 'source_hashes':sources,
        'success_rule':'No fall over 20 s and mean absolute speed error <= 0.35 m/s at a 1 m/s target.',
        'note':'Same planar quadruped topology. Three lengths vary; mass, thickness and nominal torque stay fixed. The long-upper/short-lower corner is excluded from training. Final tests use nominal dynamics. Fresh episodes, frozen weights and frozen normalization.'}
    protocol = json.loads(json.dumps(protocol))
    path = output/'report.json'
    report = read_json(path) if path.exists() else {'format':'rlsim-generalization-v1', 'protocol':protocol, 'trials':[]}
    if report['protocol'] != protocol:
        raise ValueError('Existing study uses another protocol or source revision. Choose a new output folder.')
    for seed in args.seeds:
        for enabled, condition in [(False,'fixed body'), (True,'varied bodies')]:
            key = f'{seed}-{condition}'
            if any(t['id']==key for t in report['trials']):continue
            config = TrainConfig(total_steps=args.steps, seed=seed, device='cuda',
                morphology={'enabled':enabled}, hidden_size=64, epochs=4, n_envs=8,
                rollout_steps=128,batch_size=256,evaluation_interval=131072,evaluation_episodes=3)
            run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
            folder = output/'runs'/run_id;folder.mkdir(parents=True)
            atomic_json(folder/'config.json', config.model_dump())
            atomic_json(folder/'status.json', {'status':'starting','run_id':run_id,'steps':0,'history':[],
                'config':config.model_dump(),'created_at':datetime.now(timezone.utc).isoformat()})
            report['trials'].append({'id':key,'condition':condition,'seed':seed,'run_id':run_id,
                'status':'pending','config':config.model_dump()})
    atomic_json(path,report)

    def train(trial):
        folder = output/'runs'/trial['run_id']
        if trial['status'] in ['completed','failed','trained']:return
        print('Training '+trial['id'],flush=True)
        with (folder/'worker.log').open('w') as log:
            proc = subprocess.Popen([sys.executable,'-m','morpholab.cli','worker',str(folder)],stdout=log,stderr=subprocess.STDOUT)
            code = proc.wait()
        state=read_json(folder/'status.json')
        trial.update(status='trained' if code==0 and state['status']=='completed' else 'failed',
                     steps=state.get('steps'),hardware=state.get('hardware'),training_seconds=state.get('elapsed'),
                     validation=state.get('best_validation'),error=state.get('error'))
        print(json.dumps({k:trial[k] for k in ['id','status','steps','training_seconds','error']}),flush=True)

    # Training ends before final evaluation; no test scores influence checkpoints.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(train,report['trials']):
            atomic_json(path,report)
    import torch
    torch.set_num_threads(2)
    for trial in report['trials']:
        if trial['status'] != 'trained':continue
        folder = output/'runs'/trial['run_id']
        config,model,norm = load_checkpoint(folder,'best')
        try:
            cases=[]
            for name,scales in TEST_BODIES:
                current=config.model_copy(update={'body':body_at(config.body,scales)})
                episodes=[evaluate_policy(current,model,norm,seed=seed)['summary'] for seed in TEST_SEEDS]
                cases.append({'name':name,'scales':scales,'region':region(scales),
                    'episodes':episodes,'metrics':summarize(episodes),
                    'success_rate':sum(not e['fell'] and e['speed_error']<=.35 and e['seconds']>=19.99 for e in episodes)/len(episodes)})
            trial.update(status='completed',cases=cases,checkpoint_steps=int(model.num_timesteps),
                checkpoint_sha256=hashlib.sha256((folder/'best_policy.zip').read_bytes()).hexdigest())
        except Exception as exc:
            trial.update(status='failed',error=f'Final evaluation: {type(exc).__name__}: {exc}')
        finally:norm.close()
        atomic_json(path,report)
        print('Evaluated '+trial['id'],flush=True)
    report['completed_at']=datetime.now(timezone.utc).isoformat()
    atomic_json(path,report)


if __name__=='__main__':main()
