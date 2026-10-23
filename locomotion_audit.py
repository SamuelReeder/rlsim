"""Frozen-policy diagnostics beyond the original walking gate.

Measures work from MuJoCo force/velocity at control-step endpoints (50 Hz),
so this audit is a sampled estimate, not an electrical power measurement.
Ground slip uses the velocity of the actual contacting foot-sphere surface;
non-foot floor contacts are counted separately from foot-center drift.
"""
import argparse,hashlib,itertools,json,time
from pathlib import Path
import mujoco
import numpy as np
import torch
from morpholab.environment import WalkerEnv
from morpholab.training import load_checkpoint,atomic_json
from morpholab.morphology import body_at
from morpholab.motion import walking_pass


def audit_episode(config,model,norm,scales,speed,seed):
    body=body_at(config.body,scales)
    env=WalkerEnv(body,task=config.task,environment=config.environment,target_speed=speed)
    force=np.zeros(6);velocity=np.zeros(6)
    obs,_=env.reset(seed=seed)
    work=positive=torque2=slip=contacts=nonfoot=saturation=error=0.
    start=float(env.data.qpos[0]);feet=set(int(x) for x in env._feet_ids)
    floor=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    try:
        for step in range(env.max_episode_steps):
            action,_=model.predict(norm.normalize_obs(obs.copy()),deterministic=True)
            obs,reward,terminated,truncated,info=env.step(action)
            power=env.data.actuator_force*env.data.actuator_velocity
            work+=float(np.abs(power).sum())*env.dt
            positive+=float(np.maximum(power,0).sum())*env.dt
            torque2+=float(np.mean(env.data.qfrc_actuator[3:]**2))
            saturation+=float(np.mean(np.abs(env.data.ctrl)>=.98))
            error+=abs(info['velocity']-speed)
            bad=False
            for i,c in enumerate(env.data.contact):
                if floor not in c.geom:continue
                other=int(c.geom[1] if c.geom[0]==floor else c.geom[0])
                mujoco.mj_contactForce(env.model,env.data,i,force)
                if force[0]<=.01:continue
                if other not in feet:bad=True;continue
                mujoco.mj_objectVelocity(env.model,env.data,mujoco.mjtObj.mjOBJ_GEOM,other,velocity,0)
                at_contact=velocity[3:]+np.cross(velocity[:3],c.pos-env.data.geom_xpos[other])
                slip+=float(np.linalg.norm(at_contact[:2]));contacts+=1
            nonfoot+=bad
            if terminated or truncated:break
        distance=float(env.data.qpos[0])-start
        count=step+1
        result={'steps':count,'seconds':float(env.data.time),'distance':distance,'speed_error':error/count,'fell':bool(info['fallen']),
            'gait':env.motion.summary(),'effort':env.effort.summary(distance,float(env.model.body_mass.sum()),config.environment.gravity),'physical':{'absolute_motor_work_j':work,'positive_motor_work_j':positive,
                'sampled_cost_of_transport':work/(body.body_mass*config.environment.gravity*distance) if distance>.1 else None,
                'torque_rms_nm':float(np.sqrt(torque2/count)),'saturation_fraction':saturation/count,
                'foot_surface_slip_m_s':slip/max(1,contacts),'nonfoot_ground_fraction':nonfoot/count},'seed':seed}
        result['gait_passed']=walking_pass(result)
        return result
    finally:env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--suite',choices=['corners','validation','final'],default='corners')
    args=p.parse_args();torch.set_num_threads(2)
    config,model,norm=load_checkpoint(args.source,'best')
    # Development, validation and final episode seeds are disjoint.
    seeds={'corners':[950000007],'validation':[950100007,950100021], 'final':[950200007,950200021,950200039]}[args.suite]
    if args.suite=='corners':
        cases=[{'scales':list(s),'speed':1.0} for s in itertools.product([.6,1.,1.4],repeat=3)]
        cases += [{'scales':s,'speed':v} for s in [[1,1,1],[1,.65,.65],[1,1.35,1.3],[1,1.16,.84]] for v in [.5,.75,1.25,1.5]]
    else:
        rng=np.random.default_rng(44101 if args.suite=='validation' else 44102)
        cases=[{'scales':s.tolist(),'speed':1.} for s in rng.uniform(.6,1.4,(24,3))]
        cases += [{'scales':[1,1,1],'speed':v} for v in [.5,.75,1,1.25,1.5]]
    report={'source':str(args.source),'checkpoint_steps':int(model.num_timesteps),'checkpoint_sha256':hashlib.sha256((args.source/'best_policy.zip').read_bytes()).hexdigest(),
      'suite':args.suite,'seeds':seeds,'config':config.model_dump(),'cases':[],
      'physical_metrics':'50 Hz endpoint-sampled motor work; actual MuJoCo foot-floor contact surface velocity. Not electrical power or hardware validation.'}
    start=time.monotonic()
    try:
        for case in cases:
            episodes=[audit_episode(config,model,norm,case['scales'],case['speed'],s) for s in seeds]
            report['cases'].append({**case,'episodes':episodes})
            atomic_json(args.output,report)
            print(case,'passed',sum(e['gait_passed'] for e in episodes),'/',len(episodes),'work',round(np.mean([e['physical']['absolute_motor_work_j'] for e in episodes]),1),flush=True)
        report['elapsed']=time.monotonic()-start;report['completed']=True;atomic_json(args.output,report)
    finally:norm.close()

if __name__=='__main__':main()
