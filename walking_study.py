"""Reproduce coordinated-walking runs; checkpoints selected on validation return."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import subprocess
import sys
from morpholab.training import TrainConfig, atomic_json, read_json

KINDS=('quadruped','varied')
SOURCES=('environment.py','motion.py','tasks.py','training.py','morphology.py','config.py')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1048576)
    p.add_argument('--workers',type=int,choices=[1,2],default=2)
    p.add_argument('--device',choices=['auto','cuda','cpu','mps'],default='cuda')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parent
    protocol={'source_hashes':{name:hashlib.sha256((root/'morpholab'/name).read_bytes()).hexdigest() for name in SOURCES},
              'steps':args.steps,'kinds':list(KINDS),'checkpoint':'best validation return',
              'note':'One training seed per condition, different from the original three-seed proportion study. A development comparison, not an ablation isolating any single change.'}
    protocol_path=args.output/'protocol.json'
    if protocol_path.exists() and read_json(protocol_path)!=protocol:raise ValueError('Use a new folder when source or protocol changes.')
    atomic_json(protocol_path,protocol)
    def train(kind):
        folder=args.output/kind;folder.mkdir(exist_ok=True)
        config=TrainConfig(total_steps=args.steps,seed=17 if kind=='varied' else 7,device=args.device,
            task={'preset':'walk'},morphology={'enabled':True} if kind=='varied' else None,
            hidden_size=64,epochs=4,n_envs=8,rollout_steps=128,batch_size=256,
            evaluation_interval=131072,evaluation_episodes=3,entropy_coef=.01)
        if (folder/'config.json').exists():
            old=TrainConfig.model_validate(read_json(folder/'config.json'))
            if old!=config:raise ValueError('Existing run has a different configuration.')
            if read_json(folder/'status.json').get('status')=='completed':return
            raise ValueError('Existing unfinished run; use native continuation or a new output folder.')
        atomic_json(folder/'config.json',config.model_dump())
        atomic_json(folder/'status.json',{'status':'starting','run_id':kind,'steps':0,'history':[],'config':config.model_dump()})
        with (folder/'worker.log').open('w') as log:
            proc=subprocess.run([sys.executable,'-m','morpholab.cli','worker',str(folder)],stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:raise RuntimeError(f'{kind} failed; see {folder}/worker.log')
        print(kind,'completed',flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(train,KINDS))

if __name__=='__main__':main()
