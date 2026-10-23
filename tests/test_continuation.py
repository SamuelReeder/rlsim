import os
from pathlib import Path
import pytest
from morpholab.training import TrainConfig, atomic_json, read_json, train_worker, load_checkpoint
from morpholab.continuation import read_state, digest
from morpholab.server import RunManager

@pytest.mark.skipif(not os.getenv('MORPHOLAB_NATIVE_TEST'), reason='Opt-in native PPO continuation')
@pytest.mark.parametrize('morphology',[None,{'enabled':True,'ramp_steps':4096}])
@pytest.mark.parametrize('objective',['velocity','walk'])
def test_native_continuation_keeps_optimizer_normalization_and_parent(tmp_path,morphology,objective):
    parent=tmp_path/'20261020T000000-12345678';parent.mkdir()
    config=TrainConfig(device='cpu',total_steps=4096,n_envs=2,hidden_size=32,epochs=1,
                       rollout_steps=128,batch_size=128,best_checkpoint=False,curriculum_mode='staged' if objective=='velocity' else 'speed',task={'preset':objective,'episode_seconds':1},morphology=morphology)
    atomic_json(parent/'config.json',config.model_dump())
    atomic_json(parent/'status.json',{'status':'starting','history':[],'steps':0,'created_at':'2026-10-20','config':config.model_dump()})
    train_worker(parent)
    assert read_json(parent/'status.json')['status']=='completed'
    saved=read_state(parent)
    assert saved['steps']==4096
    _, model, norm=load_checkpoint(parent)
    prior_iteration=max(int(s['step']) for s in model.policy.optimizer.state.values())
    prior_count=norm.obs_rms.count;norm.close()
    before=digest(parent/'policy.zip')
    manager=RunManager(tmp_path)
    try:
        state=manager.resume(parent.name,4096)
        child=manager.resolve(state['run_id'])
        assert child!=parent and state['start_steps']==4096
        assert manager.process.wait(timeout=60)==0
        result=read_json(child/'status.json')
        assert result['status']=='completed',result.get('error')
        assert result['steps']==8192
        continued=read_state(child)
        assert continued['schedule_steps']==4096
        if objective=='velocity':assert continued['curriculum']['index']>=saved['curriculum']['index']
        else:assert continued['curriculum'] is saved['curriculum'] is None
        _, model, norm=load_checkpoint(child)
        assert max(int(s['step']) for s in model.policy.optimizer.state.values())>prior_iteration
        assert norm.obs_rms.count>prior_count
        norm.close()
        assert digest(parent/'policy.zip')==before
    finally:manager.close()
