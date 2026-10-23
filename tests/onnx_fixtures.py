"""Generate ordinary ONNX fixtures, including external-weight files."""
from pathlib import Path
import base64

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


WEIGHTS = np.asarray([[.2, -.1], [.3, .4], [-.2, .5]], dtype=np.float32)
BIAS = np.asarray([.1, -.2], dtype=np.float32)


def create_controller(directory: Path, kind: str = "linear") -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    input_shape = [3] if kind == "vector" else [1, 3]
    output_shape = [2] if kind == "vector" else [1, 2]
    inputs = [helper.make_tensor_value_info("observations", TensorProto.FLOAT, input_shape)]
    tensors = [
        numpy_helper.from_array(WEIGHTS, "weights"),
        numpy_helper.from_array(BIAS, "bias"),
    ]
    nodes = [
        helper.make_node("MatMul", ["observations", "weights"], ["weighted"]),
        helper.make_node("Add", ["weighted", "bias"], ["actions"]),
    ]
    if kind == "multiple":
        inputs.append(helper.make_tensor_value_info("hidden_state", TensorProto.FLOAT, [1, 2]))
        nodes[-1] = helper.make_node("Add", ["weighted", "hidden_state"], ["actions"])
    elif kind == "wrong_output":
        output_shape = [1, 3]
        nodes = [helper.make_node("Identity", ["observations"], ["actions"])]
    elif kind == "nonfinite":
        tensors[1] = numpy_helper.from_array(np.asarray([np.nan, .2], dtype=np.float32), "bias")
    graph = helper.make_graph(nodes, "RLsim ONNX fixture", inputs, [
        helper.make_tensor_value_info("actions", TensorProto.FLOAT, output_shape),
    ], initializer=tensors)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    path = directory / "controller.onnx"
    if kind == "external":
        onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                        location="weights.bin", size_threshold=0)
    else:
        onnx.save_model(model, path)
    record = {"format": "rlsim-onnx-v1", "data": base64.b64encode(path.read_bytes()).decode(),
              "input_name": "observations", "output_name": "actions"}
    if kind == "external":
        record["external_data"] = [{"path": "weights.bin",
                                   "data": base64.b64encode((directory / "weights.bin").read_bytes()).decode()}]
    return record
