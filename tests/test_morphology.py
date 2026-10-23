import copy
import numpy as np
import pytest
from morpholab.config import BodyConfig
from morpholab.morphology import ProportionTraining, ProportionWalker, body_at, region, validation_scales
from morpholab.training import TrainConfig


def test_geometry_changes_without_changing_topology_or_mass():
    env=ProportionWalker(BodyConfig(),ProportionTraining(ramp_steps=0),randomize=False)
    try:
        obs,info=env.reset(seed=17)
        assert env.action_space.shape==(8,) and obs.shape==(34,)
        assert env.body.upper_length != .42
        assert np.isclose(env.model.body_mass.sum(),4)
        assert np.all(env.model.body_inertia[1:]>0)
        upper=env.model.body('upper_0').id; lower=env.model.body('lower_0').id
        assert np.isclose(env.model.body_pos[lower,2],-env.body.upper_length)
        assert env.data.qpos[1] == pytest.approx(0,abs=.1)
        assert np.allclose(obs[-12:-3],env.body.morphology_vector())
        for _ in range(30):
            obs,*_=env.step(np.zeros(8));assert np.isfinite(obs).all()
    finally:env.close()


def test_body_sampling_is_repeatable_and_excludes_withheld_region():
    env=ProportionWalker(BodyConfig(),ProportionTraining(ramp_steps=0),randomize=True)
    try:
        first,info=env.reset(seed=31)
        for _ in range(30):
            _,case=env.reset()
            assert region(case['proportions'])=='training range'
        repeated,again=env.reset(seed=31)
        np.testing.assert_array_equal(first,repeated)
        assert info['proportions']==again['proportions']
        # Restoring the saved RNG stream also restores the next sampled body.
        saved=copy.deepcopy(env.np_random.bit_generator.state)
        expected,shape=env.reset()
        env.np_random.bit_generator.state=saved
        actual,shape2=env.reset()
        np.testing.assert_array_equal(actual,expected)
        assert shape==shape2
    finally:env.close()


def test_curriculum_retained_across_body_rebuilds_and_validations_are_separate():
    env=ProportionWalker(BodyConfig(),ProportionTraining(ramp_steps=100),randomize=False)
    try:
        env.set_variation_steps(0)
        _,case=env.reset(seed=9);assert case['proportions']==[1,1,1]
        env.set_variation_steps(100);env.set_training_progress(.5)
        _,case=env.reset(seed=9)
        assert case['proportions'] != [1,1,1]
        assert env.target_speed==pytest.approx(.625)
        assert all(region(s)=='training range' for s in validation_scales())
        assert region([1,1.1,.9])=='withheld region'
        assert region([1,1.35,1.3])=='outside training range'
    finally:env.close()


@pytest.mark.parametrize('scales', [[1,1],[1,1,float('nan')],[.5,1,1],[1,1,1.5]])
def test_invalid_proportions_rejected(scales):
    with pytest.raises(ValueError):body_at(BodyConfig(),scales)


def test_invalid_training_ranges_rejected_before_start():
    with pytest.raises(ValueError):TrainConfig(body={'upper_length':.18},morphology={})
