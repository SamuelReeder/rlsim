"""Real ONNX Runtime WASM browser tests; no Python inference service.

MORPHOLAB_BROWSER_TEST=1 .venv/bin/pytest -q tests/test_onnx_browser.py
"""
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import zipfile

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.getenv("MORPHOLAB_BROWSER_TEST"),
                               reason="Opt-in real browser inference")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def browser_page():
    pw = pytest.importorskip("playwright.sync_api")
    pytest.importorskip("onnx")
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(
        QuietHandler, directory=str(ROOT / "morpholab"),
    ))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pw.sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True)
            page = browser.new_page()
            base = f"http://127.0.0.1:{server.server_port}/static/"
            # A plain text document hosts the isolated runtime without the app.
            page.goto(base + "style.css")
            page.add_script_tag(url=base + "vendor/ort/ort.webgpu.min.js")
            page.evaluate("async url => {window.ONNXPolicy = (await import(url)).ONNXPolicy;}",
                          base + "onnx-policy.mjs")
            yield page
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("kind", ["linear", "vector", "external"])
def test_conventional_onnx_executes_in_browser_wasm(browser_page, tmp_path, kind):
    from onnx_fixtures import BIAS, WEIGHTS, create_controller
    controller = create_controller(tmp_path, kind)
    observations = [[.4, -.2, .9], [1, 2, 3]]
    result = browser_page.evaluate("""async ({controller, observations}) => {
      const policy = await ONNXPolicy.restore(controller, 3, 2, {training_device:'cpu'});
      try {
        return {result: await policy.act(observations, true), device:policy.inference_device};
      } finally {await policy.dispose();}
    }""", {"controller": controller, "observations": observations})
    assert result["device"] == "cpu"
    np.testing.assert_allclose(result["result"]["actions"], np.asarray(observations) @ WEIGHTS + BIAS,
                               rtol=1e-6, atol=1e-7)
    assert result["result"]["values"] == [0, 0]


@pytest.mark.parametrize("kind,error", [
    ("multiple", "inputs"), ("wrong_output", "shape"), ("nonfinite", "nonfinite"),
])
def test_unsupported_or_invalid_controllers_fail_before_inference(browser_page, tmp_path, kind, error):
    from onnx_fixtures import create_controller
    result = browser_page.evaluate("""async controller => {
      try {const p=await ONNXPolicy.restore(controller,3,2,{training_device:'cpu'});await p.dispose();return '';}
      catch(error){return error.message;}
    }""", create_controller(tmp_path, kind))
    assert error in result.lower()


def test_onnx_shape_and_observation_validation(browser_page, tmp_path):
    from onnx_fixtures import create_controller
    result = browser_page.evaluate("""async controller => {
      const messages=[];
      try {await ONNXPolicy.restore(controller,4,2,{training_device:'cpu'});}
      catch(error){messages.push(error.message);}
      const p=await ONNXPolicy.restore(controller,3,2,{training_device:'cpu'});
      try {await p.act([[1,NaN,2]],true);}catch(error){messages.push(error.message);}
      await p.dispose();await p.dispose();
      try {await p.act([[1,2,3]],true);}catch(error){messages.push(error.message);}
      return messages;
    }""", create_controller(tmp_path))
    assert "supplies 4" in result[0]
    assert "finite" in result[1]
    assert "disposed" in result[2]


def test_example_exports_importable_onnx_package_for_custom_3d_body(browser_page, tmp_path):
    """The interface -> PyTorch example -> ZIP -> browser path keeps arbitrary bodies."""
    browser_page.add_script_tag(url="browser-client.js")
    xml = (ROOT / "examples/imports/free-body.xml").read_text()
    interface = browser_page.evaluate("""async xml => {
      window.exampleBackend = new BrowserBackend();
      const result=await exampleBackend.call('import_body',{
        files:[{path:'free-body.xml',data:new TextEncoder().encode(xml)}],
        options:{auto_actuators:true,add_floor:true},
      });
      return exampleBackend.call('interface',{
        body:result.body,environment:{gravity:5,friction:.7},
        task:{preset:'balance',episode_seconds:1,terminate_on_fall:false},
      });
    }""", xml)
    interface_path = tmp_path / "interface.json"
    interface_path.write_text(json.dumps(interface))
    output = tmp_path / "external-controller.onnx"
    try:
        completed = subprocess.run([
            sys.executable, str(ROOT / "examples/export_onnx_controller.py"),
            "--interface", str(interface_path), "--output", str(output),
        ], cwd=ROOT, capture_output=True, text=True, timeout=60)
        assert completed.returncode == 0, completed.stderr
        package = output.with_suffix(".zip")
        with zipfile.ZipFile(package) as archive:
            assert sorted(archive.namelist()) == ["external-controller.manifest.json", "external-controller.onnx"]
            manifest = json.loads(archive.read("external-controller.manifest.json"))
        assert manifest["format"] == "rlsim-onnx-manifest-v1"
        assert manifest["config"] == interface["config"]
        assert manifest["config"]["body"]["kind"] == "mjcf"
        assert manifest["observation_schema"] == "rlsim-general-v1"
        assert manifest["normalizer"] is None
        assert manifest["steps"] == 0
        result = browser_page.evaluate("""async encoded => {
          const data=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));
          const record=await exampleBackend.importModel(new File([data],'external-controller.zip'));
          const replay=await exampleBackend.request('/api/replay',{run_id:record.id});
          return {steps:record.steps,model_info:record.model_info,body:record.config.body.kind,
            schema:record.controller.observation_schema,summary:replay.summary,device:replay.inference_device};
        }""", base64.b64encode(package.read_bytes()).decode("ascii"))
        assert result["steps"] == 0
        assert result["schema"] == "rlsim-general-v1"
        assert result["body"] == "mjcf"
        assert result["model_info"]["observation_size"] == interface["model_info"]["observation_size"]
        assert result["model_info"]["action_size"] == 3
        assert result["summary"]["steps"] == 50
        assert np.isfinite(result["summary"]["reward"])
        assert result["device"] == "cpu"
    finally:
        browser_page.evaluate("exampleBackend.worker.terminate()")
