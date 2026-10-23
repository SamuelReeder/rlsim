"""Explicit reward fine-tuning fork; retains weights/Adam/normalization, resets episodes.
This is a changed-objective run, not a claim of unchanged training continuation.
"""
import argparse,hashlib,json
from pathlib import Path
import torch
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from stable_baselines3.common.monitor import Monitor
from morpholab.training import load_checkpoint,atomic_json,train_worker
from morpholab.continuation import capture
from morpholab.morphology import ProportionWalker
from walking_study import SOURCES

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--steps',type=int,default=524288);p.add_argument('--foot-drift-weight',type=float,default=1.0)
args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(2)
config,model,oldnorm=load_checkpoint(args.source,'best')
original=config.model_dump()
config=config.model_copy(update={'task':config.task.model_copy(update={'foot_drift_weight':args.foot_drift_weight}),'total_steps':args.steps,'entropy_coef':.005})
# Revalidate changed settings before writing any resumable state.
config=type(config).model_validate(config.model_dump())
envs=[ProportionWalker(config.body,proportions=config.morphology,task=config.task,target_speed=config.target_speed,environment=config.environment,randomize=config.randomize,seed=config.seed+i) for i in range(config.n_envs)]
raw=DummyVecEnv([lambda env=e:Monitor(env) for e in envs])
vec=VecNormalize.load(str(args.source/'best_normalize.pkl'),raw)
vec.training=True;vec.norm_reward=True
model.set_env(vec,force_reset=True);model.ent_coef=config.entropy_coef
model.set_random_seed(config.seed)
atomic_json(args.output/'config.json',config.model_dump())
model.save(str(args.output/'policy.zip'));vec.save(str(args.output/'normalize.pkl'))
atomic_json(args.output/'learning_state.json',capture(args.output,model,envs,None,model.num_timesteps))
atomic_json(args.output/'resume.json',{'parent_run_id':str(args.source),'start_steps':model.num_timesteps,'kind':'reward-fine-tuning'})
lineage={'kind':'reward-fine-tuning','source':str(args.source),'source_steps':model.num_timesteps,
 'source_checkpoint_sha256':hashlib.sha256((args.source/'best_policy.zip').read_bytes()).hexdigest(),
 'original_config':original,'config':config.model_dump(),'source_hashes':{n:hashlib.sha256((Path('morpholab')/n).read_bytes()).hexdigest() for n in SOURCES},
 'note':'Explicit changed-objective fork: stronger foot drift penalty and lower entropy; keeps optimizer/normalization, resets environment episodes and starts a new validation selection.'}
atomic_json(args.output/'fine_tuning.json',lineage)
atomic_json(args.output/'status.json',{'status':'starting','run_id':args.output.name,'steps':model.num_timesteps,'start_steps':model.num_timesteps,'history':[],'config':config.model_dump(),'fine_tuning':lineage})
oldnorm.close();vec.close();del model
train_worker(args.output)
