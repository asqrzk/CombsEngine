#!/usr/bin/env python3
"""Runs laya's own Agent on the states tests/laya_seat.rs asks, for parity.

This is the REAL checkpoint's reference: the tiny fixture proves the
arithmetic, and this proves that 28 layers of it, at the real rotary
bases and the real calibration, land where laya lands.

    LAYA=/path/to/laya-clone MODEL=~/.cache/combs/models/laya-typed-decisions \
      .../ref-venv/bin/python laya_reference.py out.json

laya runs f32 off CUDA — there is no autocast on this machine — so the
comparison is f32 against f32 and any gap is the port's, not a dtype's.
"""
import json
import os
import sys

sys.path.insert(0, os.environ["LAYA"])
os.environ.setdefault("USE_TF", "0")

import torch  # noqa: E402
from laya.agent import Agent  # noqa: E402

STATES = [
    "I want to cancel my subscription — please stop billing me.",
    "translate the phrase 'I'm considering unsubscription' to Arabic",
    "what's the difference between a refund and a chargeback?",
]
QUESTIONS = {
    "is_request": {
        "type": "noul",
        "instructions": "is the person asking us to do something, right now, in this message",
    },
    "intent": {
        "type": "choice",
        "instructions": "which of these is the person asking for",
        "criteria": {
            "unsubscribe": "the person is asking us to end their subscription now",
            "refund": "the person is asking for money back",
            "translate": "the person is asking us to put text into another language",
            "explain": "the person is asking what something means or how it works",
            "none": "none of these",
        },
    },
}


def main() -> None:
    agent = Agent(os.environ["MODEL"], device="cpu")
    print("device %s dtype %s" % (agent.device, agent.dtype))
    out = []
    for state in STATES:
        res = agent.system_one(state, QUESTIONS)
        row = {"state": state, "answers": {}}
        for qid, a in res["answers"].items():
            row["answers"][qid] = a
        out.append(row)
        print("\n%s" % state)
        for qid, a in res["answers"].items():
            print("  %-11s %s" % (qid, json.dumps(a)))
    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote %s" % sys.argv[1])


if __name__ == "__main__":
    main()
