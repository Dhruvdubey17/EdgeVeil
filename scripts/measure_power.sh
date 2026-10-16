set -euo pipefail

stamp="$(date +%Y%m%dT%H%M%S)"
log_dir="${TMPDIR:-/tmp}/edgeveil-power"
log="$log_dir/powermetrics-$stamp.plist"
mkdir -p "$log_dir"

export HF_HUB_OFFLINE=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1

sudo -v
sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 500 -f plist -o "$log" &
powermetrics_pid=$!
trap 'sudo kill -INT "$powermetrics_pid" 2>/dev/null || true' EXIT
sleep 5

for units in cpu-only cpu-and-gpu cpu-and-ne; do
  uv run edgeveil bench power-loop --backend coreml-ane --precision fp16 --compute-units "$units"
done
uv run edgeveil bench power-loop --backend onnxruntime --precision int8-dynamic

sudo kill -INT "$powermetrics_pid"
wait "$powermetrics_pid" || true
trap - EXIT

uv run edgeveil bench power --log "$log"
echo "powermetrics log: $log"
