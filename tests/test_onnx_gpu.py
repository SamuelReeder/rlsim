"""Opt-in real hardware ONNX WebGPU dispatch and CPU fallback verification."""
import json
import os

import numpy as np
import pytest

CDP = os.getenv("MORPHOLAB_GPU_CDP")
BASE = os.getenv("MORPHOLAB_TEST_URL", "http://localhost:8777").rstrip("/") + "/"
pytestmark = pytest.mark.skipif(not CDP, reason="Set MORPHOLAB_GPU_CDP for real ONNX GPU validation")


def test_onnx_real_webgpu_dispatch_and_cpu_fallback(tmp_path):
    pw = pytest.importorskip("playwright.sync_api")
    pytest.importorskip("onnx")
    from onnx_fixtures import BIAS, WEIGHTS, create_controller
    with pw.sync_playwright() as runtime:
        browser = runtime.chromium.connect_over_cdp(CDP)
        context = browser.new_context()
        try:
            page = context.new_page()
            page.goto(BASE + "style.css")
            page.add_script_tag(url=BASE + "vendor/ort/ort.webgpu.min.js")
            result = page.evaluate("""async ({base,controller}) => {
              const {ONNXPolicy}=await import(base+'onnx-policy.mjs');
              const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
              if(!adapter)throw new Error('A hardware WebGPU adapter is required for this test.');
              const adapterInfo=adapter.info, kernels=[];
              let dispatches=0;
              const dispatch=GPUComputePassEncoder.prototype.dispatchWorkgroups;
              GPUComputePassEncoder.prototype.dispatchWorkgroups=function(...args){
                dispatches++;return dispatch.apply(this,args);
              };
              ort.env.webgpu.profiling={mode:'default',ondata:event=>kernels.push({
                type:event.kernelType,name:event.kernelName,program:event.programName
              })};
              const gpu=await ONNXPolicy.restore(controller,3,2,{training_device:'webgpu'});
              let gpuResult;
              try {gpuResult={device:gpu.inference_device,...await gpu.act([[.4,-.2,.9]],true)};}
              finally {await gpu.dispose();}
              // Force a provider initialization error, then exercise a real
              // WASM session to verify fallback without changing browser flags.
              const original=ort.InferenceSession.create.bind(ort.InferenceSession), providers=[];
              ort.InferenceSession.create=async (data,options)=>{
                const provider=options.executionProviders[0];providers.push(provider);
                if(provider==='webgpu')throw new Error('Test unsupported GPU operator');
                return original(data,options);
              };
              let fallback;
              try {
                const cpu=await ONNXPolicy.restore(controller,3,2,{training_device:'webgpu'});
                try {fallback={device:cpu.inference_device,reason:cpu.fallback_reason,
                  ...await cpu.act([[.4,-.2,.9]],true)};}
                finally {await cpu.dispose();}
              } finally {ort.InferenceSession.create=original;}
              GPUComputePassEncoder.prototype.dispatchWorkgroups=dispatch;
              return {gpu:gpuResult,fallback,providers,kernels,dispatches,adapter:{
                vendor:adapterInfo?.vendor,architecture:adapterInfo?.architecture,
                description:adapterInfo?.description,isFallbackAdapter:adapterInfo?.isFallbackAdapter
              }};
            }""", {"base": BASE, "controller": create_controller(tmp_path)})
            expected = np.asarray([[.4, -.2, .9]]) @ WEIGHTS + BIAS
            assert result["gpu"]["device"] == "webgpu", result
            assert result["dispatches"] > 0, result
            assert not result["adapter"].get("isFallbackAdapter", False)
            np.testing.assert_allclose(result["gpu"]["actions"], expected, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(result["fallback"]["actions"], expected, rtol=1e-6, atol=1e-7)
            assert result["fallback"]["device"] == "cpu"
            assert "unsupported GPU operator" in result["fallback"]["reason"]
            assert result["providers"] == ["webgpu", "wasm"]
            print(json.dumps(result))
        finally:
            context.close()
