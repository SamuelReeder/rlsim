"""Training state beyond SB3's policy/Adam checkpoint. Episodes restart on resume."""
import hashlib
import json
import random
import numpy as np
import torch
from .training import read_json, TrainConfig

FORMAT = 'rlsim-native-training-state-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_state(path):
    state = read_json(path / 'learning_state.json')
    if state.get('format') != FORMAT:
        raise ValueError('This checkpoint has no supported training continuation state.')
    if state.get('config_signature') != config_signature(path):
        raise ValueError('The saved body or training configuration changed.')
    for name in ['policy.zip', 'normalize.pkl']:
        if state.get('files', {}).get(name) != digest(path / name):
            raise ValueError('Training state does not match the checkpoint files.')
    return state


def config_signature(path):
    # Parse defaults and numeric types before hashing: 64 and 64.0 are the
    # same physical setting, even across Pydantic serialization round-trips.
    config = TrainConfig.model_validate(read_json(path / 'config.json')).model_dump()
    for key in ['total_steps', 'device']:
        config.pop(key, None)
    if config.get('morphology') is None:
        config.pop('morphology', None)  # Existing continuation hashes remain valid.
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def capture(path, model, envs, curriculum, schedule_steps):
    numpy = np.random.get_state()
    return {'format': FORMAT, 'steps': int(model.num_timesteps), 'schedule_steps': schedule_steps, 'config_signature': config_signature(path),
            'files': {name: digest(path / name) for name in ['policy.zip', 'normalize.pkl']},
            'curriculum': {'index': curriculum.index, 'episodes': curriculum.episodes} if curriculum else None,
            'environment_random': [env.np_random.bit_generator.state for env in envs],
            'python_random': random.getstate(),
            'numpy_random': [numpy[0], numpy[1].tolist(), *numpy[2:]],
            'torch_random': torch.get_rng_state().tolist(),
            'mps_random': torch.mps.get_rng_state().tolist() if torch.backends.mps.is_available() else None,
            'cuda_random': [state.tolist() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def restore_random(state, envs):
    def tuples(value):
        return tuple(map(tuples, value)) if isinstance(value, list) else value
    random.setstate(tuples(state['python_random']))
    n = state['numpy_random']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(torch.tensor(state['torch_random'], dtype=torch.uint8))
    if torch.backends.mps.is_available() and state.get('mps_random') is not None:
        torch.mps.set_rng_state(torch.tensor(state['mps_random'], dtype=torch.uint8))
    if torch.cuda.is_available() and len(state.get('cuda_random', [])) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([torch.tensor(s, dtype=torch.uint8) for s in state['cuda_random']])
    if len(state['environment_random']) != len(envs):
        raise ValueError('Continuation environment count does not match.')
    for env, saved in zip(envs, state['environment_random']):
        env.np_random.bit_generator.state = saved
