#!/usr/bin/env python3
"""Regenerates combs-runtime/tests/data/chat_template_harmony.json.

Extracts each cached checkpoint's real chat template (GGUF metadata /
tokenizer_config.json), renders reference outputs with python-jinja2 under
transformers' environment settings (ImmutableSandboxedEnvironment,
trim_blocks + lstrip_blocks, pinned strftime_now), and writes the fixture
file the Rust harmony test compares minijinja against byte-for-byte.

Requires: pip install jinja2; the models cached under ~/.cache/combs/models.
"""
import json, os, struct
from jinja2.sandbox import ImmutableSandboxedEnvironment

PINNED_DATE = "11 Aug 2026"
BASE = os.path.expanduser("~/.cache/combs/models")
OUT = os.path.join(os.path.dirname(__file__),
                   "../../combs-runtime/tests/data/chat_template_harmony.json")

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
    "qwen3":         (gguf_chat_template(f"{BASE}/qwen3-0.6b-gguf/model.gguf"),
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

# Tool fixtures. IMPORTANT: every dict a template may `tojson` (tool
# schemas, tool_call arguments, tool_calls entries) is authored with keys
# in ALPHABETICAL order — Rust's serde_json sorts map keys, and byte
# identity requires both renderers to serialize identically. Messages
# match ChatMessage::to_template_value output exactly (content always
# present; tool_calls entries as {"function": {...}, "id": ..., "type":
# "function"}).
TOOLS = [
    {
        "function": {
            "description": "Get the current weather for a location.",
            "name": "get_weather",
            "parameters": {
                "properties": {
                    "location": {
                        "description": "City and country, e.g. Paris, France",
                        "type": "string",
                    },
                    "unit": {"description": "celsius or fahrenheit", "type": "string"},
                },
                "required": ["location"],
                "type": "object",
            },
        },
        "type": "function",
    }
]

TOOL_SETS = {
    "tools-request": [
        {"role": "user", "content": "What is the weather in Paris?"},
    ],
    "tools-loopback": [
        {"role": "user", "content": "What is the weather in Paris?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{
             "function": {"arguments": {"location": "Paris, France",
                                        "unit": "celsius"},
                          "name": "get_weather"},
             "id": "call_0",
             "type": "function",
         }]},
        {"role": "tool", "content": "22C, clear skies",
         "name": "get_weather", "tool_call_id": "call_0"},
    ],
}
TOOL_MODELS = ["llama-3.2", "qwen2.5-coder", "qwen3"]

env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
env.globals["strftime_now"] = lambda fmt: PINNED_DATE


# transformers overrides jinja2's default tojson (which HTML-escapes and
# sorts keys) with plain json.dumps; the engine's minijinja filter mirrors
# this exactly.
def tojson(x, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=False, indent=indent,
                      separators=separators, sort_keys=sort_keys)


env.filters["tojson"] = tojson

fixtures = []
for model, (tpl_src, bos, eos) in MODELS.items():
    assert tpl_src, f"{model}: no template found"
    tpl = env.from_string(tpl_src)
    for set_name, messages in SETS.items():
        # tools=None always (transformers passes it even when absent; the
        # Rust side mirrors with Value::Null).
        expected = tpl.render(messages=messages, tools=None,
                              add_generation_prompt=True,
                              bos_token=bos, eos_token=eos)
        fixtures.append({
            "name": f"{model}/{set_name}", "template": tpl_src,
            "bos_token": bos, "eos_token": eos, "date": PINNED_DATE,
            "messages": messages, "expected": expected,
        })
        print(f"OK {model}/{set_name}: {len(expected)} chars")
    if model in TOOL_MODELS:
        for set_name, messages in TOOL_SETS.items():
            expected = tpl.render(messages=messages, tools=TOOLS,
                                  add_generation_prompt=True,
                                  bos_token=bos, eos_token=eos)
            fixtures.append({
                "name": f"{model}/{set_name}", "template": tpl_src,
                "bos_token": bos, "eos_token": eos, "date": PINNED_DATE,
                "messages": messages, "tools": TOOLS, "expected": expected,
            })
            print(f"OK {model}/{set_name}: {len(expected)} chars")

json.dump(fixtures, open(OUT, "w"), indent=1)
print(f"wrote {len(fixtures)} fixtures -> {os.path.normpath(OUT)}")
