import json
import subprocess
from pathlib import Path
import numpy as np
import pytest
from morpholab.config import BodyConfig
from morpholab.environment import WalkerEnv
from morpholab.motion import MotionTracker, phase_targets, position_controls, walking_components, walking_pass


def test_standing_dragging_and_single_leg_cannot_pass():
    body=BodyConfig();tracker=MotionTracker(body)
    feet=np.zeros((4,3));tracker.reset(feet)
    for i in range(1000):
        feet[:,0]=i*.02
        tracker.update(feet,.02,0,0)
    summary=dict(seconds=20,speed_error=0,fell=False,gait=tracker.summary())
    assert summary['gait']['steps_per_leg']==[0]*4
    assert not walking_pass(summary)
    summary['gait'].update(steps_per_leg=[28,0,0,0],foot_slip_m_s=0,contact_fraction=[.5]*4)
    assert not walking_pass(summary)
    summary['gait']['steps_per_leg']=[28]*4
    assert walking_pass(summary)


def test_steps_need_airtime_clearance_and_travel():
    tracker=MotionTracker(BodyConfig());feet=np.zeros((4,3));tracker.reset(feet)
    for _ in range(10):
        for j in range(10):
            feet[0]=[j*.02,0,.06]
            feet[1]=[j*.02,0,.02]  # scraping does not count
            feet[2]=[0,0,.06]      # vertical twitch does not count
            tracker.update(feet,.02,0,0)
        feet[:,2]=0;tracker.update(feet,.02,0,0)
        feet[:]=0;tracker.update(feet,.02,0,0)
    assert tracker.summary()['steps_per_leg']==[10,0,0,0]
    tracker.reset(feet)
    assert tracker.summary()['steps_per_leg']==[0]*4


def test_gait_clock_pd_limits_and_legacy_semantics():
    e=WalkerEnv(task={'preset':'walk'})
    try:
        initial,_=e.reset(seed=13)
        assert len(initial)==e.model_info()['observation_size']==57
        assert e.model_info()['action_schema']=='rlsim-joint-targets-v1'
        assert len(e.model_info()['observation_names'])==57
        for _ in range(100):
            obs,reward,_,_,info=e.step(np.zeros(8))
            assert np.isfinite(obs).all()
            assert reward==pytest.approx(sum(info['reward_components'].values()))
            assert np.max(np.abs(e.data.ctrl))<=1
        assert not e._fallen
        assert e.motion.summary()['steps_per_leg']==[0]*4
        assert np.array_equal(initial,e.reset(seed=13)[0])
    finally:e.close()
    old=WalkerEnv()
    try:
        old.step(np.ones(8)*.2)
        assert old.observation_space.shape==(34,)
        np.testing.assert_array_equal(old.data.ctrl,np.ones(8)*.2)
    finally:old.close()


def test_python_javascript_motion_parity():
    body=BodyConfig();feet=np.array([[.1,0,.0],[.2,0,.07],[.3,0,.05],[.4,0,0.]])
    phase,contacts,heights=phase_targets(.231,.37,4,.84)
    data={'body':body.model_dump(),'feet':feet.tolist(),'phase':phase,'contacts':contacts.tolist(),'heights':heights.tolist(),
          'control':position_controls(np.arange(8)/8,np.arange(11)/20,np.arange(11)/13,body).tolist(),
          'components':walking_components(body,feet,np.array([True,False,False,True]),.231,.37,.8,1,.8,.05,.1,.2,.15,.2)}
    root=Path(__file__).resolve().parents[1]
    script='''import fs from 'node:fs';import assert from 'node:assert/strict';import {phaseTargets,positionControls,walkingComponents} from './morpholab/static/motion.mjs';
    const d=JSON.parse(fs.readFileSync(0,'utf8'));const p=phaseTargets(.231,.37,4,.84);
    const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
    close(p.phase,d.phase);assert.deepEqual(p.contacts,d.contacts);p.heights.forEach((v,i)=>close(v,d.heights[i]));
    positionControls(Array.from({length:8},(_,i)=>i/8),Array.from({length:11},(_,i)=>i/20),Array.from({length:11},(_,i)=>i/13),d.body).forEach((v,i)=>close(v,d.control[i]));
    const c=walkingComponents(d.body,d.feet,[true,false,false,true],.231,.37,.8,1,.8,.05,.1,.2,.15,.2);
    for(const [k,v] of Object.entries(c))close(v,d.components[k]);'''
    subprocess.run(['node','--input-type=module','-e',script],input=json.dumps(data),text=True,check=True,cwd=root)


def test_drift_setting_preserves_legacy_serialization_and_is_bounded():
    from pydantic import ValidationError
    from morpholab.tasks import TaskConfig
    assert 'foot_drift_weight' not in TaskConfig(preset='walk').model_dump()
    for value in [-1,6,float('nan')]:
        with pytest.raises(ValidationError):TaskConfig(preset='walk',foot_drift_weight=value)
    with pytest.raises(ValidationError):TaskConfig(preset='velocity',foot_drift_weight=1)
    body=BodyConfig();feet=np.zeros((4,3));args=(body,feet,np.ones(4,dtype=bool),.2,.1,1,1,.8,0,0,0,.5,0)
    assert walking_components(*args,foot_drift_weight=1)['foot_slip']==-.5
    assert walking_components(*args,foot_drift_weight=0)['foot_slip']==0
    env=WalkerEnv(task={'preset':'walk','foot_drift_weight':1})
    try:
        _,r,_,_,info=env.step(np.ones(8)*.2)
        assert r==pytest.approx(sum(info['reward_components'].values()))
    finally:env.close()
