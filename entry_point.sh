#!/bin/sh

# Start tailscaled in userspace networking mode
/usr/local/bin/tailscaled --tun=userspace-networking --socks5-server=localhost:1055 &
# Wait for tailscaled to be ready
sleep 3
# Authenticate and bring up Tailscale (expects TAILSCALE_AUTHKEY env var)
/usr/local/bin/tailscale up --authkey=${TAILSCALE_AUTHKEY} --hostname=surpriver --accept-routes

case "$MODE" in
  FIND_ANOMALIES)
    python3.9 detection_engine.py --top_n 25 --min_volume 5000000 --data_granularity_minutes 60 --history_to_use 14 --is_load_from_dictionary 0 --data_dictionary_path 'dictionaries/data_dict.npy' --is_save_dictionary 1 --is_test 0 --future_bars 0 --data_source 'ibgate' --output_format 'CLI'
    ;;
  FIND_CUP_N_HANDLE)
    python3.9 detection_engine.py --top_n 25 --min_volume 5000000 --data_granularity_minutes 60 --history_to_use 14 --is_load_from_dictionary 0 --data_dictionary_path 'dictionaries/data_dict.npy' --is_save_dictionary 1 --is_test 0 --future_bars 0 --data_source 'ibgate' --output_format 'CLI' --cupnhandle
    ;;
  *)
    python3.9 detection_engine.py --top_n 25 --min_volume 5000000 --data_granularity_minutes 60 --history_to_use 14 --is_load_from_dictionary 0 --data_dictionary_path 'dictionaries/data_dict.npy' --is_save_dictionary 1 --is_test 0 --future_bars 0 --data_source 'yahoo_finance' --output_format 'CLI'
    ;;
esac
