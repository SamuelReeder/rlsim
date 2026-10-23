"""Real browser learning on static hosting, including CPU fallback and storage.

MORPHOLAB_BROWSER_TEST=1 .venv/bin/pytest -q tests/test_browser_local.py
Requires Playwright Chromium. Runs a small actual PPO job, no Python RL server.
"""
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading

import pytest

pytestmark = pytest.mark.skipif(not os.getenv('MORPHOLAB_BROWSER_TEST'), reason='Opt-in real browser training')
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def static_url():
    # Serve the parent deliberately: the app must also work under a URL subpath.
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    handler = functools.partial(QuietHandler, directory=str(ROOT / 'morpholab'))
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}/static/'
    server.shutdown()
    server.server_close()
    thread.join()


def test_static_training_storage_download_import_and_inference(static_url, tmp_path):
    pw = pytest.importorskip('playwright.sync_api')
    with pw.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1280, 'height': 900})
        # Deterministically exercise unsupported-WebGPU fallback on every machine.
        def worker_without_gpu(route):
            response = route.fetch()
            route.fulfill(response=response, body="Object.defineProperty(navigator, 'gpu', {value: undefined});\n" + response.text())
        context.route('**/browser-worker.js', worker_without_gpu)
        page = context.new_page()
        errors, api_requests = [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: api_requests.append(request.url) if '/api/' in request.url else None)
        page.goto(static_url)
        page.wait_for_function("() => document.querySelector('#arena-empty').hidden", timeout=90000)
        assert page.title() == 'RLsim'
        assert page.locator('h1').inner_text() == 'RLsim'
        assert 'CPU' in page.locator('#hardware-label').inner_text()
        assert 'This device' in page.locator('#hardware-label').inner_text()
        assert 'WebGPU' in page.locator('#hardware-label').get_attribute('title')
        page.locator('#gravity').fill('8.3')
        page.locator('#friction').fill('0.7')
        page.locator('[data-tab=train]').click()
        page.locator('#advanced-training summary').click()
        advanced = {'hidden-size': 32, 'hidden-layers': 3, 'learning-rate': .0007,
                    'batch-size': 64, 'epochs': 2, 'rollout-steps': 64,
                    'discount': .97, 'entropy-coef': .01, 'clip-range': .15}
        for name, value in advanced.items():
            page.locator(f'#{name}').fill(str(value))
        page.locator('#training-budget').fill('4096')
        page.locator('#n-envs').fill('4')
        page.locator('#train-button').click()
        page.wait_for_function("() => ['Completed','Failed'].includes(document.querySelector('#run-status').textContent)", timeout=180000)
        assert page.locator('#run-status').inner_text() == 'Completed', page.locator('#status-text').inner_text()
        assert '4,096' in page.locator('#step-count').inner_text()
        page.locator('[data-tab=run]').click()
        page.wait_for_function("() => document.querySelector('#saved-run').value")
        run_id = page.locator('#saved-run').input_value()
        with page.expect_download() as download:
            page.locator('#checkpoint-link').click()
        record = json.loads(Path(download.value.path()).read_text())
        assert record['format'] == 'morpholab-browser-run-v1'
        assert record['steps'] == 4096
        expected = {name.replace('-', '_'): value for name, value in advanced.items()}
        expected['gamma'] = expected.pop('discount')
        for name, value in expected.items():
            assert record['config'][name] == value
            if name not in ('gamma', 'rollout_steps'):
                assert record['policy']['config'][name] == value
        assert record['policy']['weights']['actor_w3']['shape'] == [32, 32]
        assert record['policy']['weights']['critic_w3']['shape'] == [32, 32]
        assert record['config']['environment'] == {'gravity': 8.3, 'friction': .7}
        assert record['hardware']['training_device'] == 'cpu'
        assert len(record['history']) >= 8
        assert record['normalizer']['count'] > 4096
        assert record['metrics']['loss'] is not None
        assert len(record['policy']['weights']) > 5
        exported = tmp_path / 'model.json'
        exported.write_text(json.dumps(record))
        with page.expect_download() as xml:
            page.locator('#body-link').click()
        assert '<mujoco' in Path(xml.value.path()).read_text()
        # Browser storage survives reload; inference is computed from the saved policy.
        page.reload()
        page.wait_for_function("() => document.querySelector('#arena-empty').hidden", timeout=60000)
        page.locator('[data-tab=run]').click()
        page.wait_for_function("() => document.querySelector('#saved-run').value")
        assert page.locator('#saved-run').input_value() == run_id
        page.locator('#load-run').click()
        for name, value in advanced.items():
            assert float(page.locator(f'#{name}').input_value()) == value
        page.locator('#eval-seed').fill('237')
        page.locator('#replay-button').click()
        page.wait_for_function("() => document.querySelector('#viewer-mode').textContent === 'Model inference'", timeout=60000)
        page.wait_for_function("!document.querySelector('#evaluation-summary').hidden", timeout=60000)
        assert float(page.locator('#timeline').get_attribute('max')) > 0
        assert page.locator('#evaluation-summary').inner_text()
        assert '237' in page.locator('#status-text').inner_text()
        # Export / import on a fresh browser profile is portable and validates shapes.
        second = browser.new_context()
        second.route('**/browser-worker.js', worker_without_gpu)
        other = second.new_page()
        other.goto(static_url)
        other.wait_for_function("() => document.querySelector('#arena-empty').hidden", timeout=60000)
        other.locator('[data-tab=run]').click()
        other.locator('#model-file').set_input_files(exported)
        other.wait_for_function("() => document.querySelector('#saved-run').value", timeout=60000)
        other.locator('#replay-button').click()
        other.wait_for_function("() => document.querySelector('#viewer-mode').textContent === 'Model inference'", timeout=60000)
        other.wait_for_function("!document.querySelector('#evaluation-summary').hidden", timeout=60000)
        # Stop saves a model and releases the worker for another simulation.
        page.locator('[data-tab=train]').click()
        page.locator('#training-budget').fill('65536')
        page.locator('#train-button').click()
        page.wait_for_function("() => document.querySelector('#run-status').textContent === 'Training'", timeout=30000)
        page.locator('#stop-button').click()
        page.wait_for_function("() => ['Stopped','Failed'].includes(document.querySelector('#run-status').textContent)", timeout=60000)
        assert page.locator('#run-status').inner_text() == 'Stopped'
        assert not api_requests, f'Static mode called the server: {api_requests}'
        assert not errors
        for width in (390, 1280):
            page.set_viewport_size({'width': width, 'height': 900})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
        second.close()
        context.close()
        browser.close()
