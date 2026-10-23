"""Exercise the optional process boundary without installing SkinTokens or torch."""
import atexit
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
import threading

import pytest

from morpholab import rig_prepare as rig


def glb_bytes(*, skinned=True, external=False):
    positions = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    joints = struct.pack("<12H", *([0, 0, 0, 0] * 3))
    weights = struct.pack("<12f", *([1, 0, 0, 0] * 3))
    binary = positions + joints + weights
    document = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, 1]}],
        "nodes": [{"mesh": 0}, {"name": "root", "children": [2]}, {"name": "tip"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(joints)},
            {"buffer": 0, "byteOffset": len(positions) + len(joints), "byteLength": len(weights)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "VEC4"},
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
        ],
    }
    if skinned:
        document["nodes"][0]["skin"] = 0
        document["skins"] = [{"joints": [1, 2]}]
        document["meshes"][0]["primitives"][0]["attributes"].update(JOINTS_0=1, WEIGHTS_0=2)
    if external:
        document["images"] = [{"uri": "../texture.png"}]
    payload = json.dumps(document).encode()
    payload += b" " * (-len(payload) % 4)
    return (struct.pack("<4sII", b"glTF", 2, 28 + len(payload) + len(binary))
            + struct.pack("<I4s", len(payload), b"JSON") + payload
            + struct.pack("<I4s", len(binary), b"BIN\0") + binary)


@pytest.fixture
def installation(tmp_path, monkeypatch):
    root = tmp_path / "SkinTokens with spaces"
    for name in rig.REQUIRED_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    (root / "src/server/spec.py").write_text(f"BPY_PORT = {port}\n")
    monkeypatch.setenv("RLSIM_SKINTOKENS_HOME", str(root))
    monkeypatch.setenv("RLSIM_SKINTOKENS_PYTHON", sys.executable)
    monkeypatch.setattr(rig, "probe_runtime", lambda *_args, **_kwargs: {
        "ready": True, "issues": [], "warnings": [],
    })
    return root


@pytest.fixture
def model_paths(tmp_path):
    source, output = tmp_path / "original ; $(echo unsafe).glb", tmp_path / "results" / "prepared.glb"
    source.write_bytes(glb_bytes(skinned=False))
    return source, output


def fake_demo(root, body):
    (root / "demo.py").write_text(
        "import argparse, json, pathlib, sys, time\n"
        "parser=argparse.ArgumentParser()\n"
        "parser.add_argument('--input');parser.add_argument('--output')\n"
        "parser.add_argument('--use_transfer',action='store_true')\n"
        "parser.add_argument('--use_skeleton',action='store_true')\n"
        "args=parser.parse_args()\n"
        "pathlib.Path('invocation.json').write_text(json.dumps(vars(args)))\n"
        + body
    )


def test_quick_status_does_not_execute_external_python(installation, monkeypatch):
    monkeypatch.setattr(rig, "probe_runtime", lambda *_args, **_kwargs: pytest.fail("loaded runtime"))
    result = rig.installation_status()
    assert result["configured"] is True
    assert result["ready"] is None
    assert result["runtime_checked"] is False
    assert result["requirements"]["device"] == "NVIDIA CUDA"


def test_missing_installation_is_actionable_without_importing_ml(tmp_path):
    code = (
        "import json,sys;from morpholab.rig_prepare import installation_status;"
        "print(json.dumps(installation_status()));"
        "assert 'torch' not in sys.modules and 'bpy' not in sys.modules"
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("RLSIM_SKINTOKENS_")}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    status = json.loads(result.stdout)
    assert not status["configured"] and status["ready"] is False
    assert "RLSIM_SKINTOKENS_HOME" in " ".join(status["issues"])
    assert "RLSIM_SKINTOKENS_PYTHON" in " ".join(status["issues"])


def test_doctor_detects_missing_checkpoint_and_bad_runtime(installation, monkeypatch):
    monkeypatch.setattr(rig, "probe_runtime", lambda *_args: {
        "ready": False, "issues": ["Missing flash_attn"], "warnings": [],
    })
    status = rig.installation_status(check_runtime=True)
    assert status["configured"] and not status["ready"] and status["runtime_checked"]
    assert status["issues"] == ["Missing flash_attn"]
    (installation / rig.REQUIRED_FILES[3]).unlink()
    status = rig.installation_status()
    assert not status["configured"]
    assert "download.py --model" in status["issues"][0]


def test_dry_run_has_official_flags_and_never_creates_output(installation, model_paths):
    source, output = model_paths
    result = rig.prepare_rig(source, output, dry_run=True)
    assert result["command"] == [
        sys.executable, str(installation / "demo.py"),
        "--input", str(source), "--output", str(output), "--use_transfer",
    ]
    assert result["cwd"] == str(installation)
    assert not output.parent.exists()
    assert not result["runtime_checked"]


def test_success_runs_official_contract_preserves_source_and_publishes_rig(installation, model_paths):
    source, output = model_paths
    original = source.read_bytes()
    fake_demo(installation,
              "pathlib.Path(args.input).write_text('changed staged input')\n"
              f"pathlib.Path(args.output).write_bytes(bytes.fromhex('{glb_bytes().hex()}'))\n")
    result = rig.prepare_rig(source, output)
    assert result["status"] == "completed"
    assert source.read_bytes() == original
    assert rig.inspect_glb(output, require_rig=True)["skins"] == 1
    invoked = json.loads((installation / "invocation.json").read_text())
    assert invoked["use_transfer"] is True and invoked["use_skeleton"] is False
    assert invoked["input"] != str(source) and invoked["output"] != str(output)
    assert not Path(invoked["input"]).exists()
    assert Path(result["log"]).is_file()
    assert not (installation / ".rlsim-rig-prepare.lock").exists()


def test_skeleton_mode_requires_rig_and_forwards_only_when_requested(installation, model_paths):
    source, output = model_paths
    with pytest.raises(rig.RigPreparationError, match="Existing-skeleton"):
        rig.prepare_rig(source, output, use_skeleton=True, dry_run=True)
    source.write_bytes(glb_bytes())
    result = rig.prepare_rig(source, output, use_skeleton=True, transfer=False, dry_run=True)
    assert "--use_skeleton" in result["command"]
    assert "--use_transfer" not in result["command"]


@pytest.mark.parametrize("mode", ["same", "symlink", "exists"])
def test_original_and_existing_outputs_are_protected(installation, model_paths, mode):
    source, output = model_paths
    output.parent.mkdir()
    if mode == "same":
        output = source
    elif mode == "symlink":
        output.symlink_to(source)
    else:
        output.write_bytes(b"existing output")
    before = source.read_bytes()
    with pytest.raises(rig.RigPreparationError):
        rig.prepare_rig(source, output, overwrite=mode != "exists")
    assert source.read_bytes() == before


@pytest.mark.parametrize("payload", [b"not a GLB", glb_bytes(skinned=False)])
def test_invalid_output_does_not_replace_previous_output(installation, model_paths, payload):
    source, output = model_paths
    output.parent.mkdir()
    output.write_bytes(b"previous result")
    fake_demo(installation, f"pathlib.Path(args.output).write_bytes(bytes.fromhex('{payload.hex()}'))\n")
    with pytest.raises(rig.RigPreparationError):
        rig.prepare_rig(source, output, overwrite=True)
    assert output.read_bytes() == b"previous result"


def test_explicit_overwrite_publishes_only_on_success(installation, model_paths):
    source, output = model_paths
    output.parent.mkdir()
    output.write_bytes(b"previous result")
    fake_demo(installation, f"pathlib.Path(args.output).write_bytes(bytes.fromhex('{glb_bytes().hex()}'))\n")
    result = rig.prepare_rig(source, output, overwrite=True)
    assert result["status"] == "completed"
    assert output.read_bytes() == glb_bytes()


def test_nonzero_exit_keeps_existing_output_and_log(installation, model_paths):
    source, output = model_paths
    output.parent.mkdir()
    output.write_bytes(b"previous result")
    fake_demo(installation, "print('checkpoint could not load');sys.exit(3)\n")
    with pytest.raises(rig.RigPreparationError, match="code 3"):
        rig.prepare_rig(source, output, overwrite=True)
    assert output.read_bytes() == b"previous result"
    assert "checkpoint could not load" in next(output.parent.glob("*-rig-*.log")).read_text()


def test_cancel_event_stops_helper_and_keeps_source(installation, model_paths):
    source, output = model_paths
    original = source.read_bytes()
    fake_demo(installation, 'time.sleep(30)\n')
    cancel=threading.Event()
    with pytest.raises(rig.RigPreparationError,match='cancelled'):
        rig.prepare_rig(source,output,cancel=cancel,on_started=lambda _:cancel.set())
    assert source.read_bytes()==original and not output.exists()
    assert not (installation/'.rlsim-rig-prepare.lock').exists()


def test_raced_output_is_not_clobbered(installation, model_paths):
    source, output = model_paths
    fake_demo(installation,
              f"pathlib.Path({str(output)!r}).write_text('concurrent result')\n"
              f"pathlib.Path(args.output).write_bytes(bytes.fromhex('{glb_bytes().hex()}'))\n")
    with pytest.raises(rig.RigPreparationError, match="could not finish"):
        rig.prepare_rig(source, output)
    assert output.read_text() == "concurrent result"


def test_oversized_glb_rejected_at_browser_import_limit(tmp_path):
    source = tmp_path / "huge.glb"
    with source.open("wb") as file:
        file.truncate(rig.MAX_GLB_BYTES + 1)
    with pytest.raises(rig.RigPreparationError, match="64 MB"):
        rig.inspect_glb(source)


def test_existing_checkout_lock_is_preserved(installation, model_paths):
    lock = installation / ".rlsim-rig-prepare.lock"
    lock.write_text("1234")
    with pytest.raises(rig.RigPreparationError, match="Another rig preparation"):
        rig.prepare_rig(*model_paths)
    assert lock.read_text() == "1234"
    assert not (installation / "invocation.json").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX cancellation process contract")
def test_keyboard_cancellation_stops_owned_process(installation, model_paths):
    fake_demo(installation, "time.sleep(60)\n")
    started = {}
    def cancel(info):
        started.update(info)
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        rig.prepare_rig(*model_paths, on_started=cancel)
    with pytest.raises(ProcessLookupError):
        os.kill(started["pid"], 0)
    assert not model_paths[1].exists()
    assert not (installation / ".rlsim-rig-prepare.lock").exists()


def test_external_resources_are_rejected_before_launch(installation, model_paths):
    source, output = model_paths
    source.write_bytes(glb_bytes(external=True))
    with pytest.raises(rig.RigPreparationError, match="embedded"):
        rig.prepare_rig(source, output)
    assert not (installation / "invocation.json").exists()


def test_occupied_blender_port_is_never_reused_or_stopped(installation, model_paths):
    with socket.socket() as service:
        service.bind(("127.0.0.1", rig._bpy_port(installation)))
        service.listen()
        status = rig.installation_status()
        assert status["configured"] and status["ready"] is False
        with pytest.raises(rig.RigPreparationError, match="already in use"):
            rig.prepare_rig(*model_paths)
        assert service.fileno() >= 0
        assert not (installation / "invocation.json").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX detached-child cleanup contract")
def test_timeout_cleans_detached_blender_child_and_keeps_original(installation, model_paths):
    source, output = model_paths
    before = source.read_bytes()
    fake_demo(installation,
        "import subprocess, signal, os\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)\n"
        "pathlib.Path('child.pid').write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    with pytest.raises(rig.RigPreparationError, match="timed out"):
        rig.prepare_rig(source, output, timeout=.3)
    assert source.read_bytes() == before
    assert not output.exists()
    assert not (installation / ".rlsim-rig-prepare.lock").exists()
    pid = int((installation / "child.pid").read_text())
    for _ in range(20):
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        if not state or state.startswith("Z"):
            break
        time.sleep(.05)
    assert not state or state.startswith("Z"), "Detached child survived cancellation"


def test_cli_doctor_and_server_status_are_read_only(installation, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["rlsim", "rig-prepare", "--doctor"])
    from morpholab.cli import main
    main()
    from fastapi.testclient import TestClient
    from morpholab.server import create_app
    app = create_app(tmp_path / "runs")
    try:
        with TestClient(app) as client:
            response = client.get("/api/rig-prepare/status")
            assert response.status_code == 200
            assert response.json()["configured"] is True
            assert response.json()["runtime_checked"] is False
            assert client.post("/api/rig-prepare/status", json={}).status_code == 405
    finally:
        atexit.unregister(app.state.manager.close)
