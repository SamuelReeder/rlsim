"""Optional, isolated SkinTokens preparation. This module imports no ML runtime."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import tempfile
import time
from typing import Callable


UPSTREAM = "https://github.com/VAST-AI-Research/SkinTokens"
UPSTREAM_REVISION = "273b691d35989d71cd17ff2895fdc735097b92d1"
REQUIRED_FILES = (
    "demo.py", "bpy_server.py", "src/server/spec.py",
    "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
    "experiments/skin_vae_2_10_32768/last.ckpt",
    "models/Qwen3-0.6B/config.json",
)
MAX_GLB_BYTES = 64 * 1024 * 1024
RUNTIME_MARKER = "RLSIM_RIG_RUNTIME="
RUNTIME_PROBE = r"""
import importlib.util, json, sys
result = {"python_version": list(sys.version_info[:3]), "issues": [], "warnings": []}
if sys.version_info < (3, 11):
    result["issues"].append("SkinTokens requires Python 3.11 or newer.")
modules = ["torch", "bpy", "flash_attn", "transformers", "diffusers", "box", "einops",
           "omegaconf", "lightning", "addict", "fast_simplification", "trimesh",
           "open3d", "huggingface_hub", "numpy", "scipy", "rtree", "gradio", "bottle", "tornado"]
missing = [name for name in modules if importlib.util.find_spec(name) is None]
if missing:
    result["issues"].append("Missing modules in the SkinTokens interpreter: " + ", ".join(missing))
if "torch" not in missing:
    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        if not result["cuda_available"]:
            result["issues"].append("SkinTokens needs an NVIDIA CUDA GPU; CPU and MPS are unsupported.")
        else:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            free, total = torch.cuda.mem_get_info(device)
            result.update(gpu_name=props.name, total_memory_bytes=total, free_memory_bytes=free,
                          torch_cuda_version=torch.version.cuda)
            cuda_version = tuple(int(part) for part in (torch.version.cuda or "0.0").split(".")[:2])
            if cuda_version < (12, 1):
                result["issues"].append("SkinTokens requires CUDA 12.1 or newer in its PyTorch runtime.")
            if total < 14 * 1024**3:
                result["issues"].append("SkinTokens requires an NVIDIA GPU with at least 14 GB of memory.")
            if free < 14 * 1024**3:
                result["warnings"].append("Less than 14 GB of GPU memory is currently free. Stop other GPU workloads before preparing a rig.")
            probe = torch.ones(8, device="cuda")
            (probe * probe).sum().item()
            torch.cuda.synchronize()
            del probe
        for name in ("bpy", "flash_attn"):
            if name not in missing:
                try:
                    __import__(name)
                except Exception as error:
                    result["issues"].append(name + " could not load: " + str(error))
    except Exception as error:
        result["issues"].append("CUDA runtime check failed: " + str(error))
result["ready"] = not result["issues"]
print("RLSIM_RIG_RUNTIME=" + json.dumps(result))
"""


class RigPreparationError(RuntimeError):
    """A preparation failed before a validated output was published."""


def _installation(checkout=None, python=None) -> tuple[Path | None, Path | None]:
    checkout = checkout or os.environ.get("RLSIM_SKINTOKENS_HOME")
    python = python or os.environ.get("RLSIM_SKINTOKENS_PYTHON")
    root = Path(checkout).expanduser().resolve() if checkout else None
    # Keep the executable symlink: resolving .venv/bin/python would bypass its venv.
    executable = Path(python).expanduser().absolute() if python else None
    return root, executable


def _bpy_port(root: Path) -> int:
    try:
        tree = ast.parse((root / "src/server/spec.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "BPY_PORT" for target in node.targets
            ):
                value = ast.literal_eval(node.value)
                if type(value) is int and 1 <= value <= 65535:
                    return value
    except (OSError, SyntaxError, ValueError):
        pass
    raise RigPreparationError("Cannot determine BPY_PORT from SkinTokens src/server/spec.py.")


def _port_available(port: int) -> bool:
    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def probe_runtime(executable: Path, root: Path, timeout: float = 60) -> dict:
    """Explicitly inspect the separate environment; never run during quick status."""
    try:
        result = subprocess.run(
            [str(executable), "-c", RUNTIME_PROBE], cwd=root, capture_output=True,
            text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ready": False, "issues": [f"SkinTokens runtime check failed: {exc}"], "warnings": []}
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(RUNTIME_MARKER):
            try:
                value = json.loads(line[len(RUNTIME_MARKER):])
                if result.returncode:
                    value["ready"] = False
                    value["issues"].append(f"Runtime probe exited with code {result.returncode}.")
                return value
            except (ValueError, TypeError, KeyError):
                break
    return {"ready": False, "issues": [
        "SkinTokens Python did not finish its runtime check. " + result.stderr[-2000:].strip(),
    ], "warnings": []}


def installation_status(checkout=None, python=None, *, check_runtime: bool = False) -> dict:
    root, executable = _installation(checkout, python)
    issues = []
    if root is None:
        issues.append("Set RLSIM_SKINTOKENS_HOME or pass --checkout for an external SkinTokens checkout.")
    else:
        missing = [name for name in REQUIRED_FILES if not (root / name).is_file() or not (root / name).stat().st_size]
        if missing:
            issues.append("Missing SkinTokens files: " + ", ".join(missing)
                          + ". Follow the upstream setup and run its download.py --model.")
    if executable is None:
        issues.append("Set RLSIM_SKINTOKENS_PYTHON or pass --python for its separate Python environment.")
    elif not executable.is_file() or not os.access(executable, os.X_OK):
        issues.append(f"SkinTokens Python is not executable: {executable}")
    configured = not issues
    port = None
    if root is not None and (root / "src/server/spec.py").is_file():
        try:
            port = _bpy_port(root)
            if not _port_available(port):
                issues.append(f"SkinTokens Blender port {port} is already in use. Stop that service yourself before preparing a rig.")
        except RigPreparationError as exc:
            issues.append(str(exc))
    result = {
        "feature": "SkinTokens / TokenRig", "configured": configured,
        "ready": False if issues else None, "runtime_checked": False,
        "issues": issues, "warnings": [], "checkout": str(root) if root else None,
        "python": str(executable) if executable else None, "blender_port": port,
        "upstream": UPSTREAM, "reference_revision": UPSTREAM_REVISION,
        "requirements": {"python": ">=3.11", "device": "NVIDIA CUDA", "minimum_gpu_memory_gb": 14},
        "command": "rlsim rig-prepare --doctor --check-runtime",
        "execution_location": "local", "input_formats": [".glb"], "output_format": ".glb",
    }
    if check_runtime and not issues:
        runtime = probe_runtime(executable, root)
        result.update(runtime=runtime, ready=bool(runtime["ready"]), runtime_checked=True)
        result["issues"].extend(runtime["issues"])
        result["warnings"].extend(runtime.get("warnings", []))
    return result


def inspect_glb(path: Path, *, require_rig: bool = False, require_skin: bool = False) -> dict:
    """Check container integrity and that a publishable rig actually has skinning."""
    size = path.stat().st_size
    if size < 20 or size > MAX_GLB_BYTES:
        raise RigPreparationError("GLB must be a nonempty glTF 2.0 binary file no larger than RLsim's 64 MB import limit.")
    with path.open("rb") as source:
        magic, version, length = struct.unpack("<4sII", source.read(12))
        if magic != b"glTF" or version != 2 or length != size:
            raise RigPreparationError("Invalid GLB header or length.")
        chunk_length, chunk_type = struct.unpack("<I4s", source.read(8))
        if chunk_type != b"JSON" or chunk_length > min(size - 20, 16 * 1024 * 1024):
            raise RigPreparationError("GLB needs a JSON header smaller than 16 MB.")
        try:
            document = json.loads(source.read(chunk_length))
        except (ValueError, UnicodeError) as exc:
            raise RigPreparationError("Invalid GLB JSON header.") from exc
        if (not isinstance(document, dict) or not isinstance(document.get("asset"), dict)
                or document["asset"].get("version") != "2.0"):
            raise RigPreparationError("GLB must declare glTF 2.0.")
        while source.tell() < size:
            header = source.read(8)
            if len(header) != 8:
                raise RigPreparationError("Truncated GLB chunk.")
            count, _ = struct.unpack("<I4s", header)
            if count > size - source.tell():
                raise RigPreparationError("Truncated GLB data.")
            source.seek(count, 1)
    for key in ("buffers", "images", "nodes", "skins", "meshes", "accessors"):
        if not isinstance(document.get(key, []), list) or not all(
            isinstance(item, dict) for item in document.get(key, [])
        ):
            raise RigPreparationError(f"Invalid GLB {key} list.")
    for resource in [*document.get("buffers", []), *document.get("images", [])]:
        uri = resource.get("uri", "")
        if not isinstance(uri, str) or (uri and not uri.startswith("data:")):
            raise RigPreparationError("Use a self-contained GLB with embedded textures and buffers.")
    nodes, skins, meshes = (document.get(name, []) for name in ("nodes", "skins", "meshes"))
    accessors = document.get("accessors", [])
    def valid_attribute(primitive, name, shape):
        index = primitive.get("attributes", {}).get(name)
        return (type(index) is int and 0 <= index < len(accessors)
                and accessors[index].get("type") == shape
                and type(accessors[index].get("count")) is int and accessors[index]["count"] > 0)
    for mesh in meshes:
        if not isinstance(mesh.get("primitives"), list) or not all(
            isinstance(primitive, dict) and isinstance(primitive.get("attributes"), dict)
            for primitive in mesh["primitives"]
        ):
            raise RigPreparationError("Invalid GLB mesh primitives.")
    if not any(valid_attribute(primitive, "POSITION", "VEC3")
               for mesh in meshes for primitive in mesh["primitives"]):
        raise RigPreparationError("GLB must contain a mesh with vertex positions.")
    if require_rig or require_skin:
        if not skins or not all(
            isinstance(skin.get("joints"), list) and skin["joints"]
            and all(type(index) is int and 0 <= index < len(nodes) for index in skin["joints"])
            for skin in skins
        ):
            raise RigPreparationError("GLB has no valid skeleton/skin. Existing-skeleton mode requires a rigged input.")
    if require_rig:
        bound_meshes = [node.get("mesh") for node in nodes
                        if type(node.get("skin")) is int and 0 <= node["skin"] < len(skins)]
        if not any(
            type(index) is int and 0 <= index < len(meshes)
            and any(valid_attribute(primitive, "POSITION", "VEC3")
                    and valid_attribute(primitive, "JOINTS_0", "VEC4")
                    and valid_attribute(primitive, "WEIGHTS_0", "VEC4")
                    for primitive in meshes[index].get("primitives", []))
            for index in bound_meshes
        ):
            raise RigPreparationError("SkinTokens output has no mesh with joint indices and skin weights.")
    return {"bytes": size, "nodes": len(nodes), "skins": len(skins), "meshes": len(meshes)}


def build_command(executable: Path, root: Path, source: Path, output: Path, *,
                  use_skeleton: bool = False, transfer: bool = True) -> list[str]:
    command = [str(executable), str(root / "demo.py"), "--input", str(source), "--output", str(output)]
    if transfer:
        command.append("--use_transfer")
    if use_skeleton:
        command.append("--use_skeleton")
    return command


@contextmanager
def _checkout_lock(root: Path):
    path = root / ".rlsim-rig-prepare.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RigPreparationError(
            f"Another rig preparation owns {path}. If its process has exited, remove this stale lock."
        ) from exc
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def _descendants(pid: int) -> set[int]:
    """Include upstream Blender's separate process group when cancelling."""
    if os.name == "nt":
        return set()
    try:
        listing = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True, text=True, timeout=3, check=True)
        rows = [tuple(map(int, line.split())) for line in listing.stdout.splitlines()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return set()
    children, parents = set(), {pid}
    while parents:
        parents = {child for child, parent in rows if parent in parents and child not in children}
        children.update(parents)
    return children


def _terminate(process: subprocess.Popen) -> None:
    children = _descendants(process.pid)
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        try:
            # Python handles SIGINT by unwinding, allowing upstream atexit cleanup.
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            children.update(_descendants(process.pid))
        for pid in children:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        # Detached Blender children may ignore SIGTERM while inside native code.
        for pid in children:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    process.wait(timeout=5)


def prepare_rig(input_path, output_path, *, checkout=None, python=None,
                use_skeleton: bool = False, transfer: bool = True, overwrite: bool = False,
                timeout: float = 1800, dry_run: bool = False,
                on_started: Callable[[dict], None] | None = None,
                cancel=None, preserve_mesh: bool = False,
                on_stage: Callable[[str], None] | None = None) -> dict:
    if not math.isfinite(timeout) or not 0 < timeout <= 86400:
        raise RigPreparationError("Timeout must be between 0 and 86400 seconds.")
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().absolute()
    if source.suffix.lower() != ".glb" or not source.is_file():
        raise RigPreparationError("Input must be an existing self-contained .glb file.")
    if destination.suffix.lower() != ".glb":
        raise RigPreparationError("Output must have a .glb extension.")
    if destination.is_symlink() or destination.resolve() == source:
        raise RigPreparationError("Output must be a different regular path; the original is never overwritten.")
    if destination.exists() and (not overwrite or not destination.is_file()):
        raise RigPreparationError("Output already exists. Choose a new path or pass --overwrite.")
    inspect_glb(source, require_skin=use_skeleton)
    if preserve_mesh:
        from .rig_fidelity import read_glb, world_matrices, mesh_points
        try:
            document, binary = read_glb(source)
            if any('skin' in node for node in document.get('nodes', [])):
                raise ValueError('This model already has a skeleton. Import with Rig: Auto to preserve it.')
            world, _ = world_matrices(document)
            if not list(mesh_points(document, binary, world)):
                raise ValueError('The GLB contains no usable mesh.')
        except (ValueError, KeyError, IndexError) as exc:
            raise RigPreparationError(str(exc)) from exc
    if on_stage:
        on_stage('checking')
    status = installation_status(checkout, python, check_runtime=not dry_run)
    if status["issues"]:
        raise RigPreparationError(" ".join(status["issues"]))
    if cancel is not None and cancel.is_set():
        raise RigPreparationError('Rig preparation cancelled.')
    root, executable = Path(status["checkout"]), Path(status["python"])
    if dry_run:
        return {**status, "dry_run": True, "input": str(source), "output": str(destination),
                "cwd": str(root), "command": build_command(executable, root, source, destination,
                    use_skeleton=use_skeleton, transfer=transfer),
                "note": "No inference ran. Execution uses temporary copies and publishes output after validation."}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _checkout_lock(root):
        if not _port_available(status["blender_port"]):
            raise RigPreparationError(f"SkinTokens Blender port {status['blender_port']} is already in use.")
        with tempfile.TemporaryDirectory(prefix=".rlsim-rig-", dir=destination.parent) as temporary:
            work = Path(temporary)
            copied_input, staged_output = work / "input.glb", work / "prepared.glb"
            # A standalone GLB can be copied without losing linked dependencies.
            import shutil
            shutil.copyfile(source, copied_input)
            descriptor, log_name = tempfile.mkstemp(prefix=destination.stem + "-rig-", suffix=".log", dir=destination.parent)
            log_path = Path(log_name)
            command = build_command(executable, root, copied_input, staged_output,
                                    use_skeleton=use_skeleton, transfer=transfer)
            started = time.monotonic()
            process = None
            try:
                with os.fdopen(descriptor, "w") as log:
                    process = subprocess.Popen(
                        command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=os.name != "nt",
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                    )
                    if on_started:
                        on_started({"pid": process.pid, "log": str(log_path), "warnings": status["warnings"]})
                    if on_stage:
                        on_stage('rigging')
                    try:
                        while process.poll() is None:
                            if cancel is not None and cancel.is_set():
                                raise KeyboardInterrupt()
                            if time.monotonic() - started > timeout:
                                raise subprocess.TimeoutExpired(command, timeout)
                            time.sleep(.2)
                        code = process.returncode
                    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                        _terminate(process)
                        reason = "cancelled" if isinstance(exc, KeyboardInterrupt) else f"timed out after {timeout:g} seconds"
                        raise RigPreparationError(f"Rig preparation {reason}; original and destination preserved. Log: {log_path}") from exc
                if code:
                    raise RigPreparationError(f"SkinTokens exited with code {code}. See {log_path}")
                if not staged_output.is_file() or staged_output.is_symlink():
                    raise RigPreparationError(f"SkinTokens produced no regular GLB output. See {log_path}")
                info = inspect_glb(staged_output, require_rig=True)
                fidelity = None
                if preserve_mesh:
                    if on_stage:
                        on_stage('verifying')
                    from .rig_fidelity import preserve_original
                    preserved_output = work / 'preserved.glb'
                    try:
                        fidelity = preserve_original(source, staged_output, preserved_output)
                    except (ValueError, KeyError, IndexError) as exc:
                        raise RigPreparationError(f'Model preservation check failed: {exc}') from exc
                    staged_output = preserved_output
                    info = inspect_glb(staged_output, require_rig=True)
                if cancel is not None and cancel.is_set():
                    raise RigPreparationError('Rig preparation cancelled.')
                if overwrite:
                    if destination.is_symlink() or destination.resolve() == source:
                        raise RigPreparationError("Output path changed during preparation; refusing to overwrite.")
                    os.replace(staged_output, destination)
                else:
                    # Atomic no-clobber publication, including concurrent writers.
                    os.link(staged_output, destination)
                return {"status": "completed", "input": str(source), "output": str(destination),
                        "log": str(log_path), "elapsed": time.monotonic() - started,
                        "model_info": info, "warnings": status["warnings"],
                        **({'fidelity': fidelity} if fidelity else {})}
            except OSError as exc:
                raise RigPreparationError(f"Rig preparation could not finish: {exc}. Log: {log_path}") from exc
            finally:
                if process is not None and process.poll() is None:
                    _terminate(process)
