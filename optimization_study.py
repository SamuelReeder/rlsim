"""Paired refinement: broader coverage with/without mechanical-work shaping."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,subprocess,sys
from morpholab.training import atomic_json,read_json,TrainConfig


def prepare(source,folder,work_weight,seed,steps):
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
    from stable_baselines3.common.monitor import Monitor
    from morpholab.training import load_checkpoint
    from morpholab.continuation import capture
    from morpholab.morphology import ProportionWalker,ProportionTraining
    torch.set_num_threads(2)
    config,model,oldnorm=load_checkpoint(source,'best')
    original=config.model_dump()
    config=TrainConfig.model_validate({**original,'total_steps':steps,'seed':seed,'entropy_coef':.003,
        'task':{**original['task'],'mechanical_work_weight':work_weight,'foot_only_support':True},
        'morphology':{'enabled':True,'width':.4,'withhold_width':.2,'ramp_steps':2000000,
                      'speed_range':[.5,1.5],'boundary_fraction':.2,'robust_validation':True},
        'evaluation_episodes':8,'evaluation_interval':131072})
    envs=[ProportionWalker(config.body,proportions=config.morphology,task=config.task,
          environment=config.environment,target_speed=config.target_speed,randomize=True,seed=seed+i) for i in range(config.n_envs)]
    raw=DummyVecEnv([lambda e=e:Monitor(e) for e in envs])
    vec=VecNormalize.load(str(source/'best_normalize.pkl'),raw);vec.training=True;vec.norm_reward=True
    model.set_env(vec,force_reset=True);model.ent_coef=config.entropy_coef;model.set_random_seed(seed)
    atomic_json(folder/'config.json',config.model_dump());model.save(str(folder/'policy.zip'));vec.save(str(folder/'normalize.pkl'))
    atomic_json(folder/'learning_state.json',capture(folder,model,envs,None,int(model.num_timesteps)))
    atomic_json(folder/'resume.json',{'parent_run_id':str(source),'start_steps':int(model.num_timesteps),'kind':'coverage-and-efficiency-refinement'})
    atomic_json(folder/'status.json',{'status':'starting','run_id':folder.name,'steps':int(model.num_timesteps),
        'start_steps':int(model.num_timesteps),'history':[],'config':config.model_dump(),'parent_run_id':str(source),
        'fine_tuning':{'changes':'Broader proportions and speed commands; foot-only support; measured mechanical-work penalty.',
                       'mechanical_work_weight':work_weight,'original_config':original}})
    oldnorm.close();vec.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('experiments/gait-refinement/varied'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=1048576)
    p.add_argument('--weights',type=float,nargs='+',default=[0,.2]);p.add_argument('--seed',type=int,default=37)
    p.add_argument('--workers',type=int,default=2,choices=[1,2])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    files=['environment.py','effort.py','motion.py','morphology.py','tasks.py','training.py','config.py']
    protocol={'source':str(args.source),'source_checkpoint_sha256':hashlib.sha256((args.source/'best_policy.zip').read_bytes()).hexdigest(),
      'source_hashes':{n:hashlib.sha256((Path('morpholab')/n).read_bytes()).hexdigest() for n in files},
      'weights':args.weights,'seed':args.seed,'additional_steps':args.steps,'created_at':datetime.now(timezone.utc).isoformat(),
      'checkpoint_selection':'Mean return across eight separate validation bodies/speeds; final audit never used for checkpoint choice.',
      'conditions':'Same pretrained weights, seed, budget, physics and sampling. Only mechanical-work weight differs. Not independent from-scratch training seeds.'}
    atomic_json(args.output/'protocol.json',protocol)
    folders=[]
    for weight in args.weights:
        folder=args.output/('work-'+str(weight));folder.mkdir();prepare(args.source,folder,weight,args.seed,args.steps);folders.append(folder)
    def run(folder):
        with (folder/'worker.log').open('w') as log:
            proc=subprocess.run([sys.executable,'-m','morpholab.cli','worker',str(folder)],stdout=log,stderr=subprocess.STDOUT)
        state=read_json(folder/'status.json');print(folder.name,proc.returncode,state['status'],state.get('steps'),flush=True)
        if proc.returncode or state['status']!='completed':raise RuntimeError(f'Failed: {folder}')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(run,folders))
    protocol['completed_at']=datetime.now(timezone.utc).isoformat();atomic_json(args.output/'protocol.json',protocol)

if __name__=='__main__':main()
