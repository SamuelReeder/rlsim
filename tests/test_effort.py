import copy
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import pytest
from pydantic import ValidationError
from morpholab.config import BodyConfig
from morpholab.effort import EffortMeter
from morpholab.environment import WalkerEnv
from morpholab.morphology import ProportionTraining, ProportionWalker, validation_cases
from morpholab.tasks import TaskConfig
from morpholab.training import TrainConfig


def test_work_signs_transport_normalization_and_js_parity():
    meter=EffortMeter()
    assert meter.summary(0,4,9.81)['mechanical_cost_of_transport'] is None
    samples=[([2,-3],[4,5],[12,-18],[.99,-.5],.004),([1,2],[-3,4],[6,12],[.1,-1],.004)]
    work=[meter.observe(*args) for args in samples]
    assert sum(work)==pytest.approx(.136)
    summary=meter.summary(2,4,10)
    assert summary['positive_motor_work_j']==pytest.approx(.064)
    assert summary['mean_motor_power_w']==pytest.approx(17)
    assert summary['mechanical_cost_of_transport']==pytest.approx(.0017)
    assert summary['torque_rms_nm']==pytest.approx(np.sqrt(162))
    assert summary['actuator_saturation_fraction']==.5
    script="""import fs from 'node:fs';import {EffortMeter} from './morpholab/static/effort.mjs';
    const m=new EffortMeter();for(const args of JSON.parse(fs.readFileSync(0,'utf8')))m.observe(...args);
    console.log(JSON.stringify(m.summary(2,4,10)));"""
    out=subprocess.check_output(['node','--input-type=module','-e',script],input=json.dumps(samples),text=True,cwd=Path(__file__).resolve().parents[1])
    assert json.loads(out)==pytest.approx(summary)
    meter.reset();assert meter.summary(2,4,10)['absolute_motor_work_j']==0


def test_work_matches_motor_power_and_does_not_change_legacy_physics():
    env=WalkerEnv(randomize=False)
    try:
        env.reset(seed=25)
        reference=mujoco.MjData(env.model)
        mujoco.mj_copyData(reference,env.model,env.data)
        action=np.linspace(-.3,.4,8)
        for _ in range(20):
            reference.ctrl[:]=action
            mujoco.mj_step(env.model,reference,nstep=5)
            mujoco.mj_forward(env.model,reference)
            _,_,_,_,info=env.step(action)
            np.testing.assert_array_equal(env.data.qpos,reference.qpos)
            np.testing.assert_array_equal(env.data.qvel,reference.qvel)
            assert np.sum(env.data.actuator_force*env.data.actuator_velocity)==pytest.approx(np.dot(env.data.qfrc_actuator,env.data.qvel))
            assert info['motor_work_j']>=0
        assert env.effort.summary(2,4,9.81)['absolute_motor_work_j']>0
        env.reset();assert env.effort.summary(2,4,9.81)['absolute_motor_work_j']==0
    finally:env.close()


def test_foot_only_support_and_work_penalty_use_real_physics():
    env=WalkerEnv(task={'preset':'walk','foot_only_support':True,'mechanical_work_weight':.2,'min_height_ratio':0,'max_tilt_degrees':180})
    try:
        env.reset(seed=13)
        for _ in range(30):
            _,reward,done,_,info=env.step(np.zeros(8))
            assert not info['nonfoot_contact'] and not done
            assert reward==pytest.approx(sum(info['reward_components'].values()))
            expected=-.2*info['motor_work_j']/(env.model.body_mass.sum()*9.81*env.target_speed*env.dt)
            assert info['reward_components']['mechanical_work']==pytest.approx(expected)
        env.data.qpos[1]=.05-env.model.body_pos[env._torso_id,2]
        mujoco.mj_forward(env.model,env.data)
        _,reward,done,_,info=env.step(np.zeros(8))
        assert info['nonfoot_contact'] and info['fallen'] and done
        assert info['reward_components']['nonfoot_contact']==-2
        assert reward==pytest.approx(sum(info['reward_components'].values()))
        assert 'gait_contacts' not in info['reward_components']
    finally:env.close()


def test_optional_fields_preserve_old_serialization_and_reject_invalid_values():
    task=TaskConfig(preset='walk').model_dump()
    assert 'mechanical_work_weight' not in task and 'foot_only_support' not in task
    for value in [-1,3,float('nan')]:
        with pytest.raises(ValidationError):TaskConfig(preset='walk',mechanical_work_weight=value)
    for value in ['yes',1]:
        with pytest.raises(ValidationError):TaskConfig(preset='walk',foot_only_support=value)
    with pytest.raises(ValidationError):TaskConfig(preset='velocity',foot_only_support=True)
    assert ProportionTraining().model_dump()==dict(enabled=True,width=.2,ramp_steps=131072)
    for data in [{'width':.5},{'width':.2,'withhold_width':.3},{'speed_range':[2,1]},{'speed_range':[.1,float('inf')]},{'boundary_fraction':.8}]:
        with pytest.raises(ValidationError):ProportionTraining(**data)


def test_broad_sampling_reproducibility_and_withheld_corner():
    m=ProportionTraining(width=.4,withhold_width=.2,speed_range=(.5,1.5),boundary_fraction=.5,robust_validation=True,ramp_steps=0)
    env=ProportionWalker(BodyConfig(),m,task={'preset':'walk'},randomize=True)
    try:
        env.reset(seed=99);boundary=0;speeds=[]
        for _ in range(80):
            _,case=env.reset();s=np.array(case['proportions'])
            assert np.all(s>=.6) and np.all(s<=1.4)
            assert not (s[1]>=1.1 and s[2]<=.9)
            boundary+=bool(np.allclose(np.abs(s-1),.4))
            speeds.append(env.requested_target_speed)
        assert boundary>10 and min(speeds)<.6 and max(speeds)>1.4
        saved=copy.deepcopy(env.np_random.bit_generator.state)
        obs,case=env.reset();speed=env.requested_target_speed
        env.np_random.bit_generator.state=saved
        again,again_case=env.reset()
        np.testing.assert_array_equal(obs,again);assert case==again_case and speed==env.requested_target_speed
        cfg=TrainConfig(task={'preset':'walk'},morphology=m)
        for case in validation_cases(cfg):
            assert not (case.body.upper_length/cfg.body.upper_length>=1.1 and case.body.lower_length/cfg.body.lower_length<=.9)
    finally:env.close()


def test_falls_are_reported_even_when_termination_is_disabled():
    from morpholab.training import evaluate_policy
    config=TrainConfig(task={'preset':'walk','episode_seconds':1,'terminate_on_fall':False,'min_height_ratio':1,'foot_only_support':True})
    r=evaluate_policy(config,capture=True)
    assert r['summary']['steps']==50
    assert any(f['fallen'] for f in r['frames'])
    assert r['summary']['fell'] is True
