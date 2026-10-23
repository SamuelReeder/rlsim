import os
import numpy as np
import pytest
from morpholab.curriculum import LearningCurriculum
from morpholab.training import TrainConfig, atomic_json, read_json, train_worker, load_checkpoint, replay_run
from morpholab.environment import WalkerEnv


def test_staged_curriculum_requires_success_and_preserves_task_schema():
    c=LearningCurriculum(1)
    for _ in range(12):
        c.record(fell=True,speed_error=0)
    assert not c.advance()
    for index in range(3):
        for _ in range(12):
            c.record(fell=False,speed_error=0)
        assert c.advance() and c.index == index+1
    config=TrainConfig(curriculum_mode="staged")
    assert config.task.preset == "velocity"
    with pytest.raises(ValueError,match="Velocity"):
        TrainConfig(curriculum_mode="staged",task={"preset":"custom"})
    env=WalkerEnv(config.body,task=config.task,randomize=True)
    try:
        shape=env.observation_space.shape
        env.set_learning_stage(LearningCurriculum(1).stage)
        obs,_=env.reset(seed=7)
        assert obs.shape == shape and env.target_speed == 0 and env.task.preset == "balance"
        np.testing.assert_array_equal(env._domain_scales,[1,1,1])
    finally:
        env.close()


@pytest.mark.skipif(not os.getenv("MORPHOLAB_NATIVE_TEST"),reason="Opt-in real native PPO job")
def test_native_best_checkpoint_staged_training_and_replay(tmp_path):
    config=TrainConfig(device="cpu",total_steps=8192,n_envs=2,hidden_size=32,
                       epochs=2,rollout_steps=128,batch_size=128,curriculum_mode="staged",
                       evaluation_interval=4096,evaluation_episodes=2,task={"episode_seconds":1})
    atomic_json(tmp_path/"config.json",config.model_dump())
    atomic_json(tmp_path/"status.json",{"status":"starting","history":[],"steps":0,"config":config.model_dump()})
    train_worker(tmp_path)
    status=read_json(tmp_path/"status.json")
    assert status["status"] == "completed", status.get("error")
    best=read_json(tmp_path/"best.json")
    assert best["steps"] in [4096,8192]
    assert len(best["validation"]["episodes"]) == 2
    assert status["curriculum"]["name"] in ["Balance","Slow walk","Walk","Target speed"]
    assert (tmp_path/"best_policy.zip").is_file()
    assert (tmp_path/"best_normalize.pkl").is_file()
    counts=[]
    for selection in ["best","latest"]:
        _,model,normalizer=load_checkpoint(tmp_path,selection)
        counts.append(normalizer.obs_rms.count)
        assert model.target_kl == .03
        normalizer.close()
        result=replay_run(tmp_path,checkpoint=selection,seed=400007)
        assert result["checkpoint"] == selection
        assert result["summary"]["steps"] > 0
        assert np.isfinite(result["summary"]["return"])
    assert counts[0]<=counts[1]
