# Runs on a fresh runpod/base pod: installs the Concrete-ML stack, runs JOBS one by one,
# prints each result to the pod log. Env: JOBS ("dataset seed variant requests" lines),
# PATCH_B64 (tar.gz of files not on the pinned commit), MAX_S, JOB_TIMEOUT.
set -u
echo "BOOT start $(date -u +%H:%M:%S)"
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket"; nproc; free -g | sed -n 2p
lscpu | grep -o -E "avx512f|avx2" | sort -u | tr '\n' ' '; echo
which runpodctl; env | grep -c RUNPOD_API_KEY
( sleep "${MAX_S:-32400}"; echo "WATCHDOG stop"; runpodctl pod stop "$RUNPOD_POD_ID" 2>&1 | tail -n 2 ) &
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null && apt-get install -y -qq binutils >/dev/null; echo "BOOT apt $?"
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; export PATH=$HOME/.local/bin:$PATH
mkdir -p /work && cd /work
curl -LsSf https://github.com/ekinboztepe/cipherforest/archive/57b95ac.tar.gz | tar xz --strip-components=1
echo "$PATCH_B64" | base64 -d | tar xz
uv venv -q -p 3.11 /venv && . /venv/bin/activate
uv pip install -q "torch==2.3.1" --index-url https://download.pytorch.org/whl/cpu; echo "BOOT torch $?"
uv pip install -q "concrete-ml==1.9.0"; echo "BOOT cml $?"
uv pip install -q --no-deps -e .; echo "BOOT repo $?"
python -c "import concrete.ml; print('BOOT import ok')"
echo "BOOT ready $(date -u +%H:%M:%S)"
cd /work/experiments
echo "$JOBS" | while read -r d s v r; do
  [ -z "$d" ] && continue
  echo "JOB start $d $s $v $r $(date -u +%H:%M:%S)"
  timeout "${JOB_TIMEOUT:-2400}" python server_bench.py "$d" "$s" "$v" "$r" 2>&1 | grep -E "^RESULT|Error|Killed" | cut -c1-20000
  echo "JOB end $d $s $v exit ${PIPESTATUS[0]} $(date -u +%H:%M:%S)"
done
echo "ALL DONE $(date -u +%H:%M:%S)"
runpodctl pod stop "$RUNPOD_POD_ID" 2>&1 | tail -n 2; echo "STOP attempted"
sleep infinity
