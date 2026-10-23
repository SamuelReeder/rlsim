"""Real imported-body training on static hosting; no server-side physics."""
import json
import os
from pathlib import Path
import subprocess

import pytest
from test_browser_local import static_url, ROOT

pytestmark = pytest.mark.skipif(not os.getenv('MORPHOLAB_BROWSER_TEST'), reason='Opt-in real browser imports')


def test_imported_body_task_training_portable_checkpoint_and_character(static_url, tmp_path):
    pw = pytest.importorskip('playwright.sync_api')
    # The tiny character has a real skin and joint hierarchy, not an animation.
    rig = tmp_path / 'character.glb'
    subprocess.run(['node', '--input-type=module', '-e',
        "import {tinyRigGLB} from './tests/scene-fixtures.mjs'; import {writeFileSync} from 'node:fs'; writeFileSync(process.argv[1],new Uint8Array(tinyRigGLB()));", str(rig)], cwd=ROOT, check=True)
    with pw.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1400, 'height': 1000})
        def without_gpu(route):
            response = route.fetch()
            route.fulfill(response=response, body="Object.defineProperty(navigator,'gpu',{value:undefined});\n" + response.text())
        context.route('**/browser-worker.js', without_gpu)
        page = context.new_page()
        errors, api_requests = [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: api_requests.append(request.url) if '/api/' in request.url else None)
        page.goto(static_url)
        page.wait_for_function("document.querySelector('#arena-empty').hidden", timeout=90000)
        page.locator('#asset-files').set_input_files(str(ROOT / 'examples/imports/free-body.xml'))
        page.wait_for_function("!document.querySelector('#body-import-options').hidden")
        page.locator('#apply-body-import').click()
        page.wait_for_function("!document.querySelector('#imported-body-info').hidden", timeout=30000)
        assert int(page.locator('#actuator-count').inner_text()) > 0
        assert page.locator('#body-dimensions').is_hidden()
        assert page.locator('#physics-label').inner_text() == '3D physics'
        page.locator('[data-tab=train]').click()
        page.locator('#task-settings > summary').click()
        page.locator('#task-preset').select_option('custom')
        page.locator('#episode-seconds').fill('1')
        page.locator('#terminate-on-fall').uncheck()
        page.locator('#reward-expression').fill('1 + 0.1*vy - 0.01*abs(vz) + 0.01*energy')
        page.locator('#advanced-training summary').click()
        for key, value in {'training-budget':4096, 'n-envs':2, 'hidden-size':16, 'hidden-layers':1, 'epochs':1, 'rollout-steps':64}.items():
            page.locator('#'+key).fill(str(value))
        page.locator('#train-button').click()
        page.wait_for_function("['Completed','Failed'].includes(document.querySelector('#run-status').textContent)", timeout=180000)
        assert page.locator('#run-status').inner_text() == 'Completed', page.locator('#status-text').inner_text()
        page.locator('[data-tab=run]').click()
        page.wait_for_function("document.querySelector('#saved-run').value")
        with page.expect_download() as download:
            page.locator('#checkpoint-link').click()
        record = json.loads(Path(download.value.path()).read_text())
        assert record['config']['body']['kind'] == 'mjcf'
        assert record['config']['task']['preset'] == 'custom'
        assert record['model_info']['observation_schema'] == 'rlsim-general-v1'
        assert record['normalizer']['count'] > 4000
        exported = tmp_path / 'controller.json'
        exported.write_text(json.dumps(record))
        page.reload()
        page.wait_for_function("document.querySelector('#arena-empty').hidden", timeout=60000)
        page.locator('[data-tab=run]').click()
        page.locator('#model-file').set_input_files(str(exported))
        page.wait_for_function("document.querySelector('#status-text').textContent === 'Model imported'", timeout=60000)
        page.locator('#replay-button').click()
        page.wait_for_function("!document.querySelector('#evaluation-summary').hidden", timeout=90000)
        replay = page.evaluate('state.frames')
        assert replay[-1]['time'] == pytest.approx(1, abs=.021)
        assert replay[-1]['observation_schema'] == 'rlsim-general-v1'
        assert len(replay[0]['bodies']) > 1
        page.locator('[data-tab=body]').click()
        page.locator('#asset-files').set_input_files(str(rig))
        page.wait_for_function("document.querySelector('#import-entry').value.endsWith('character.glb')")
        page.locator('#apply-body-import').click()
        page.wait_for_function("document.querySelector('#status-text').textContent === 'Body imported on This device'", timeout=60000)
        character = page.evaluate('state.body')
        assert character['visual']['mode'] == 'skinned'
        assert len(character['visual']['bone_bodies']) >= 2
        assert int(page.locator('#actuator-count').inner_text()) >= 3
        # Body import replaces geometry; it never silently rewrites the character as a preset.
        assert character['metadata']['source_format'] in ('gltf','glb')
        assert not errors, errors
        assert not api_requests, api_requests
        browser.close()


def test_onnx_package_interface_validation_and_saved_inference(static_url, tmp_path):
    import zipfile
    import numpy as np
    onnx = pytest.importorskip('onnx')
    from onnx import helper, TensorProto, numpy_helper
    pw = pytest.importorskip('playwright.sync_api')
    with pw.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        context = browser.new_context()
        def without_gpu(route):
            response = route.fetch()
            route.fulfill(response=response, body="Object.defineProperty(navigator,'gpu',{value:undefined});\n" + response.text())
        context.route('**/browser-worker.js', without_gpu)
        page = context.new_page()
        errors, requests = [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: requests.append(request.url) if '/api/' in request.url else None)
        page.goto(static_url)
        page.wait_for_function("document.querySelector('#arena-empty').hidden", timeout=60000)
        interface = page.evaluate("""async()=>browserBackend.call('interface',{
          body:state.body,environment:readEnvironment(),task:{preset:'balance',episode_seconds:1,terminate_on_fall:false}
        })""")
        info = interface['model_info']
        obs, actions = info['observation_size'], info['action_size']
        assert len(info['observation_names']) == obs
        assert len(info['action_names']) == actions
        model = helper.make_model(helper.make_graph([
            helper.make_node('MatMul', ['observations','weights'], ['actions'])], 'portable controller',
            [helper.make_tensor_value_info('observations',TensorProto.FLOAT,[1,obs])],
            [helper.make_tensor_value_info('actions',TensorProto.FLOAT,[1,actions])],
            [numpy_helper.from_array(np.zeros((obs,actions),dtype=np.float32),'weights')]),
            opset_imports=[helper.make_opsetid('',17)], ir_version=8)
        manifest = {**interface['manifest_template'], 'model':'controller.onnx',
                    'input_name':'observations','output_name':'actions','name':'External test controller'}
        package = tmp_path / 'controller.zip'
        with zipfile.ZipFile(package,'w') as archive:
            archive.writestr('bundle/controller.onnx',model.SerializeToString())
            archive.writestr('bundle/manifest.json',json.dumps(manifest))
        page.locator('[data-tab=run]').click()
        page.locator('#model-file').set_input_files(str(package))
        page.wait_for_function("document.querySelector('#status-text').textContent === 'Model imported'",timeout=60000)
        saved = page.evaluate('async()=>browserBackend.records(state.selectedRun)')
        assert saved['controller']['format'] == 'rlsim-onnx-v1'
        assert saved['controller']['observation_schema'] == 'rlsim-planar-task-v1'
        assert saved['inference_device'] == 'cpu'
        page.locator('#replay-button').click()
        page.wait_for_function("!document.querySelector('#evaluation-summary').hidden", timeout=60000)
        assert page.evaluate('state.frames.at(-1).time') == pytest.approx(1,abs=.021)
        page.reload()
        page.wait_for_function("document.querySelector('#arena-empty').hidden",timeout=60000)
        page.locator('[data-tab=run]').click()
        page.locator('#replay-button').click()
        page.wait_for_function("!document.querySelector('#evaluation-summary').hidden",timeout=60000)
        assert page.evaluate('state.frames.at(-1).time') == pytest.approx(1,abs=.021)
        # Asset filenames cannot overwrite canonical MJCF in a portable body ZIP.
        archive = page.evaluate('''async()=>{
          let result;
          const record={body_xml:'<mujoco model="canonical"/>',config:{body:{kind:'mjcf',assets:[
            {path:'body.xml',data:btoa('<mujocoinclude/>')},
            {path:'rlsim-body-1.xml',data:btoa('<mujocoinclude/>')}]}}};
          await BrowserBackend.prototype.exportModel.call({records:async()=>record,
            download:(data,name)=>{result={data:Array.from(data),name}}},'test','body');
          return result;
        }''')
        import io
        with zipfile.ZipFile(io.BytesIO(bytes(archive['data']))) as bundle:
            assert bundle.read('rlsim-body-2.xml') == b'<mujoco model="canonical"/>'
            assert bundle.read('body.xml') == b'<mujocoinclude/>'
            assert bundle.read('rlsim-body-1.xml') == b'<mujocoinclude/>'
        # An incompatible schema fails before adding a second library record.
        saved['controller']['observation_schema']='not-the-saved-schema'
        message=page.evaluate("async record=>{try{await browserBackend.call('validate',record);return ''}catch(e){return e.message}}",saved)
        assert 'schema' in message
        assert not errors, errors
        assert not requests, requests
        browser.close()
