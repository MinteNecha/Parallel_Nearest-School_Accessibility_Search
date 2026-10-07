#!/usr/bin/env bash
# Functional tests. Oversubscribed ranks are used here for correctness only, never for timing.
set -u
cd "$(dirname "$0")/.."
fail=0; n=0
check() {
    out=$(mpiexec -n "$1" ./nsearch --synth "$2" --pool "$3" --alg "$4" --strategy "$5" \
          --k "$6" --q "$7" --B "$8" --trials 1 --verify --label t 2>&1 | grep '^t,' | head -1)
    n=$((n + 1))
    [ "${out##*,}" = "1" ] || { echo "FAIL alg=$4 strat=$5 p=$1 k=$6 q=$7 B=$8: $out"; fail=1; }
}
for alg in dijkstra bfs; do
  for strat in source graph scenario; do
    for p in 1 2 3 4; do
      for B in 1 7; do check $p 5000 250 $alg $strat 37 501 $B; done
    done
    for cfg in "3 1 1" "2 1 3" "1 4 5" "5 3 2"; do check 4 2000 60 $alg $strat $cfg; done
  done
done
echo "parallel-vs-sequential configurations checked: $n"
python3 -I tests/test_pipeline.py || fail=1
[ $fail = 0 ] && echo "ALL TESTS PASSED" || { echo "TESTS FAILED"; exit 1; }
