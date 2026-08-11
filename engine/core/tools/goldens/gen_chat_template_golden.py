#!/usr/bin/env python3
"""Regenerates combs-runtime/tests/data/chat_template_golden.json.

Extracts each cached checkpoint's real chat template (GGUF metadata /
tokenizer_config.json), renders reference outputs with python-jinja2 under
transformers' environment settings (ImmutableSandboxedEnvironment,
trim_blocks + lstrip_blocks, pinned strftime_now), and writes the fixture
file the Rust golden test compares minijinja against byte-for-byte.

Requires: pip install jinja2; the models cached under ~/.cache/combs/models.
"""
import json, os, struct
from jinja2.sandbox import ImmutableSandboxedEnvironment

PINNED_DATE = "11 Aug 2026"
BASE = os.path.expanduser("~/.cache/combs/models")
OUT = os.path.join(os.path.dirname(__file__),
                   "../../combs-runtime/tests/data/chat_template_golden.json")

def gguf_chat_template(path):
    f = open(path, "rb")
    u32 = lambda: struct.unpack("<I", f.read(4))[0]
    u64 = lambda: struct.unpack("<Q", f.read(8))[0]
    def s():
        n = u64(); return f.read(n).decode("utf-8", "replace")
    assert f.read(4) == b"GGUF"
    u32(); u64(); n_kv = u64()
    def skip(t):
        if t in (0,1,7): f.read(1)
        elif t in (2,3): f.read(2)
        elif t in (4,5,6): f.read(4)
        elif t in (10,11,12): f.read(8)
        elif t == 8: s()
        elif t == 9:
            et = u32(); n = u64()
            for _ in range(n): skip(et)
    for _ in range(n_kv):
        key = s(); t = u32()
        if key == "tokenizer.chat_template" and t == 8:
            return s()
        skip(t)
    return None

MODELS = {
    "llama-3.2":     (gguf_chat_template(f"{BASE}/llama-3.2-1b-instruct-gguf/model.gguf"),
                      "<|begin_of_text|>", "<|eot_id|>"),
    "qwen2.5-coder": (gguf_chat_template(f"{BASE}/qwen2.5-coder-7b-instruct-gguf/model.gguf"),
                      "<|endoftext|>", "<|im_end|>"),
    "gemma-3":       (json.load(open(f"{BASE}/gemma-3-1b-it/tokenizer_config.json"))["chat_template"],
                      "<bos>", "<end_of_turn>"),
    "smollm2":       (json.load(open(f"{BASE}/smollm2-360m/tokenizer_config.json"))["chat_template"],
                      "<|im_start|>", "<|im_end|>"),
}

SETS = {
    "sys-user": [
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "List three primes."},
    ],
    "multi-turn": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Write a haiku about bees."},
    ],
    "bare-user": [
        {"role": "user", "content": "Explain mmap briefly."},
    ],
}

env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
env.globals["strftime_now"] = lambda fmt: PINNED_DATE

fixtures = []
for model, (tpl_src, bos, eos) in MODELS.items():
    assert tpl_src, f"{model}: no template found"
    tpl = env.from_string(tpl_src)
    for set_name, messages in SETS.items():
        expected = tpl.render(messages=messages, add_generation_prompt=True,
                              bos_token=bos, eos_token=eos)
        fixtures.append({
            "name": f"{model}/{set_name}", "template": tpl_src,
            "bos_token": bos, "eos_token": eos, "date": PINNED_DATE,
            "messages": messages, "expected": expected,
        })
        print(f"OK {model}/{set_name}: {len(expected)} chars")

json.dump(fixtures, open(OUT, "w"), indent=1)
print(f"wrote {len(fixtures)} fixtures -> {os.path.normpath(OUT)}")
