#!/usr/bin/env python3
"""Tiny ONNX fixtures for the container-reader tests.

Builds three artifacts + an expectations JSON:
- plain.onnx            — fp32/fp16/int64 initializers, raw_data inline
- external.onnx (+ .data) — the same weights moved to an external-data
                            file with explicit (offset, length) entries
- nbits.onnx            — one com.microsoft.MatMulNBits node with
                            packed int4 weight + scales initializers

Deterministic contents (sin patterns, never RNG). The Rust side
(combs-formats/tests/onnx_reader.rs, gated on COMBS_ONNX_FIXTURE_DIR)
parses each file and checks names, dims, dtypes, byte contents and the
MatMulNBits attributes against expected.json.

Usage:
  ~/.cache/combs/ref-venv/bin/python tools/harmony/gen_onnx_fixture.py <out-dir>
"""
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.external_data_helper import convert_model_to_external_data

OUT = sys.argv[1] if len(sys.argv) > 1 else "onnx-fixtures"
os.makedirs(OUT, exist_ok=True)


def det(shape, scale=1.0, phase=0.0, dtype=np.float32):
    n = int(np.prod(shape))
    x = np.sin(np.arange(n, dtype=np.float64) * 0.0137 + phase) * scale
    return x.reshape(shape).astype(dtype)


expected = {"plain": {}, "external": {}, "nbits": {}}

# ---- plain: three dtypes, raw inline ---------------------------------
w_f32 = det((4, 8), 1.0)
w_f16 = det((3, 5), 0.5, phase=1.0, dtype=np.float16)
w_i64 = np.arange(6, dtype=np.int64).reshape(2, 3)

inits = [
    numpy_helper.from_array(w_f32, name="model.layers.0.mlp.weight"),
    numpy_helper.from_array(w_f16, name="model.layers.0.attn.weight"),
    numpy_helper.from_array(w_i64, name="model.rope.positions"),
]
x = helper.make_tensor_value_info("input_ids", TensorProto.FLOAT, [1, 8])
y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4])
node = helper.make_node("MatMul", ["input_ids", "model.layers.0.mlp.weight"], ["logits"])
graph = helper.make_graph([node], "plain", [x], [y], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
onnx.save(model, os.path.join(OUT, "plain.onnx"))

for arr, name in [(w_f32, "model.layers.0.mlp.weight"), (w_f16, "model.layers.0.attn.weight"), (w_i64, "model.rope.positions")]:
    expected["plain"][name] = {
        "dims": list(arr.shape),
        "dtype": str(arr.dtype),
        "bytes_hex": arr.tobytes().hex(),
    }
expected["plain"]["graph_inputs"] = ["input_ids"]
expected["plain"]["graph_outputs"] = ["logits"]

# ---- external: same graph, weights spilled with offsets --------------
model_ext = onnx.load(os.path.join(OUT, "plain.onnx"))
convert_model_to_external_data(
    model_ext,
    all_tensors_to_one_file=True,
    location="external.onnx.data",
    size_threshold=0,
    convert_attribute=False,
)
onnx.save(model_ext, os.path.join(OUT, "external.onnx"))
ext = onnx.load(os.path.join(OUT, "external.onnx"), load_external_data=False)
for init in ext.graph.initializer:
    entries = {e.key: e.value for e in init.external_data}
    expected["external"][init.name] = {
        "location": entries.get("location"),
        "offset": int(entries.get("offset", "0")),
        "length": int(entries["length"]),
    }

# ---- nbits: one MatMulNBits with packed int4 + scales ----------------
K, N, BLOCK = 64, 8, 32
blocks_per_col = K // BLOCK
packed = det((N, blocks_per_col, BLOCK // 2), 60.0, phase=2.0).astype(np.int8).astype(np.uint8)
scales = det((N * blocks_per_col,), 0.02, phase=3.0)
b_init = numpy_helper.from_array(packed, name="qweight")
s_init = numpy_helper.from_array(scales, name="qscales")
a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, K])
out = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, N])
nb = helper.make_node(
    "MatMulNBits",
    ["A", "qweight", "qscales"],
    ["Y"],
    name="qmm",
    domain="com.microsoft",
    K=K,
    N=N,
    bits=4,
    block_size=BLOCK,
)
graph = helper.make_graph([nb], "nbits", [a], [out], [b_init, s_init])
model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
)
onnx.save(model, os.path.join(OUT, "nbits.onnx"))
expected["nbits"] = {
    "K": K,
    "N": N,
    "bits": 4,
    "block_size": BLOCK,
    "inputs": ["A", "qweight", "qscales"],
    "qweight_bytes_hex": packed.tobytes().hex(),
    "qscales_bytes_hex": scales.tobytes().hex(),
}

with open(os.path.join(OUT, "expected.json"), "w") as f:
    json.dump(expected, f, indent=1)
print(f"[onnx-fixtures] wrote plain/external/nbits -> {OUT}")
