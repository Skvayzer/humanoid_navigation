#!/usr/bin/env bash
set -u
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

duration_s="${1:-1800}"
interval_s="${2:-2}"
log_dir="$G1_SLAM_RUNTIME_DIR/logs"
mkdir -p "$log_dir"
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
output="$log_dir/drift_monitor_${run_id}.tsv"
fast_log="$log_dir/fast_lio.log"
end_epoch=$(( $(date +%s) + duration_s ))

printf 'time\tload1\tmem_available_kb\tmax_temp_mc\tfastlio_pid\tfastlio_cpu\tfastlio_rss_kb\trosbridge_cpu\tlivox_driver_cpu\tlivox_viz_cpu\tno_effective_count\tnet_rx_drop\tnet_tx_drop\n' > "$output"

while (( $(date +%s) < end_epoch )); do
  now="$(date --iso-8601=seconds)"
  load1="$(awk '{print $1}' /proc/loadavg)"
  mem_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  max_temp="$(awk 'BEGIN {m=0} {if ($1>m) m=$1} END {print m}' /sys/class/thermal/thermal_zone*/temp 2>/dev/null)"

  fast_pid="$(pgrep -n -f '/fast_lio/fastlio_mapping' 2>/dev/null || true)"
  if [[ -n "$fast_pid" ]]; then
    read -r fast_cpu fast_rss < <(ps -p "$fast_pid" -o %cpu=,rss= 2>/dev/null || printf '0 0\n')
  else
    fast_cpu=0
    fast_rss=0
  fi

  rosbridge_cpu="$(ps -eo %cpu=,args= | awk '/rosbridge_websocket/ && !/awk/ {sum+=$1} END {print sum+0}')"
  livox_driver_cpu="$(ps -eo %cpu=,args= | awk '/livox_ros_driver2_node/ && !/awk/ {sum+=$1} END {print sum+0}')"
  livox_viz_cpu="$(ps -eo %cpu=,args= | awk '/livox_viz_node/ && !/awk/ {sum+=$1} END {print sum+0}')"
  no_effective="$(grep -c 'No Effective Points' "$fast_log" 2>/dev/null || true)"
  read -r rx_drop tx_drop < <(awk -F'[: ]+' 'NR>2 {rx+=$6; tx+=$14} END {print rx+0, tx+0}' /proc/net/dev)

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$now" "$load1" "$mem_available" "${max_temp:-0}" "${fast_pid:-0}" \
    "$fast_cpu" "$fast_rss" "$rosbridge_cpu" "$livox_driver_cpu" \
    "$livox_viz_cpu" "$no_effective" "$rx_drop" "$tx_drop" >> "$output"
  sleep "$interval_s"
done

echo "$output"
