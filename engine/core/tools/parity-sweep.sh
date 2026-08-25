#!/bin/sh
# Per-architecture decode parity, run the only safe way: ONE TEST PER
# PROCESS. Each test loads its own engine; the GPU pool never returns
# memory until the process exits, and autotune's working set scales
# super-linearly with vocabulary (measured: 20.5 GB peak for one
# qwen3-0.6b test, 3.2 GB for smollm2-360m). Running a suite in one
# process stacks those spikes and has taken a whole machine down.
#
#   tools/parity-sweep.sh [model.gguf ...]
#
# With no arguments, sweeps the cached models known to fit comfortably.
# Large-vocab models beyond qwen3-0.6b (gemma3's 262k vocab, anything
# >1B) are deliberately not in the default list: estimate first with the
# combs-formats footprint probe, and run with the machine quiet.
set -e
cd "$(dirname "$0")/.."

MODELS="$@"
if [ -z "$MODELS" ]; then
  MODELS="$HOME/.cache/combs/models/smollm2-360m-instruct-gguf/model.gguf \
          $HOME/.cache/combs/models/qwen3-0.6b-gguf/model.gguf"
fi
TESTS="greedy_transcript bytes_loaded_model_matches_file_loaded local_engine_matches_threaded_engine unseeded_sampled_generation_runs"

CARGO_INCREMENTAL=0 cargo test --release -p combs-runtime --test decode_transcript --no-run >/dev/null 2>&1
BIN=$(CARGO_INCREMENTAL=0 cargo test --release -p combs-runtime --test decode_transcript --no-run 2>&1 \
      | grep -o 'target/release/deps/decode_transcript-[a-f0-9]*' | head -1)

FAIL=0
for m in $MODELS; do
  [ -f "$m" ] || { echo "SKIP (missing): $m"; continue; }
  echo "== $(basename $(dirname "$m"))"
  for t in $TESTS; do
    if COMBS_TEST_GGUF="$m" "./$BIN" --ignored --test-threads=1 --exact "$t" 2>&1 \
        | grep -q "test result: ok. 1 passed"; then
      echo "   ok   $t"
    else
      echo "   FAIL $t"; FAIL=1
    fi
  done
done
[ "$FAIL" -eq 0 ] && echo "PARITY SWEEP: clean" || echo "PARITY SWEEP: FAILURES"
exit $FAIL
