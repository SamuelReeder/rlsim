import threading
import time

from fastapi.testclient import TestClient
import pytest

from morpholab import rig_jobs, rig_prepare
from morpholab.server import create_app
from test_rig_prepare import glb_bytes


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(rig_prepare, 'installation_status', lambda: {'configured': True})
    app = create_app(tmp_path / 'runs')
    with TestClient(app) as client:
        yield client
    app.state.rig_jobs.close()


def wait(client, token):
    for _ in range(100):
        state = client.get(f'/api/rig-prepare/jobs/{token}').json()
        if state['status'] in {'completed','cancelled','failed'}:
            return state
        time.sleep(.01)
    pytest.fail('Rig job did not finish')


def test_upload_isolated_job_preserves_source_and_returns_result(client, monkeypatch):
    def prepare(source, output, **kwargs):
        assert source.read_bytes() == glb_bytes(skinned=False)
        assert kwargs['preserve_mesh'] is True
        kwargs['on_stage']('verifying')
        output.write_bytes(glb_bytes())
        return {'fidelity': {'joints': 2}, 'elapsed': .1, 'warnings': []}
    monkeypatch.setattr(rig_jobs, 'prepare_rig', prepare)
    response = client.post('/api/rig-prepare/jobs', content=glb_bytes(skinned=False), headers={'Content-Type':'model/gltf-binary'})
    assert response.status_code == 202
    token = response.json()['id']
    assert wait(client, token)['status'] == 'completed'
    assert client.get(f'/api/rig-prepare/jobs/{token}/model').content == glb_bytes()
    assert client.app.state.rig_jobs.root.joinpath(token,'input.glb').read_bytes() == glb_bytes(skinned=False)


def test_cancellation_and_single_job_limit(client, monkeypatch):
    def prepare(source, output, **kwargs):
        kwargs['cancel'].wait(3)
        raise RuntimeError('cancelled')
    monkeypatch.setattr(rig_jobs, 'prepare_rig', prepare)
    request = {'content': glb_bytes(skinned=False), 'headers': {'Content-Type':'model/gltf-binary'}}
    token = client.post('/api/rig-prepare/jobs', **request).json()['id']
    assert client.post('/api/rig-prepare/jobs', **request).status_code == 409
    assert client.get(f'/api/rig-prepare/jobs/{token}/model').status_code == 409
    assert client.post(f'/api/rig-prepare/jobs/{token}/cancel').status_code == 200
    assert wait(client, token)['status'] == 'cancelled'
    assert client.get(f'/api/rig-prepare/jobs/{token}/model').status_code == 409


def test_bad_upload_and_cross_origin_requests_do_not_start_helper(client, monkeypatch):
    monkeypatch.setattr(rig_jobs, 'prepare_rig', lambda *a, **kw: pytest.fail('Unexpected subprocess'))
    assert client.post('/api/rig-prepare/jobs', content=b'bad',headers={'Content-Type':'model/gltf-binary'}).status_code == 422
    assert client.post('/api/rig-prepare/jobs', content=b'bad').status_code == 415
    assert client.post('/api/rig-prepare/jobs', headers={'Origin':'https://unrelated.example'}).status_code == 403
    assert client.post('/api/rig-prepare/jobs', headers={'Host':'public.example'}).status_code == 403
    assert client.get('/api/rig-prepare/jobs/not-a-job').status_code == 404
