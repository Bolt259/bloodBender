#!/bin/bash
echo "=== Sync Progress Monitor ==="
echo ""
echo "Process Status:"
ps aux | grep "bloodbank_download.py" | grep -v grep | awk '{print "  PID: "$2" - CPU: "$3"% - MEM: "$4"% - "$11" "$12" "$13" "$14}'
echo ""
echo "Pump 881235 Progress (last 5 lines):"
tail -5 /home/bolt/projects/bb/bloodbank_pump_881235_full.log 2>/dev/null | grep "Processing chunk" | tail -1
echo ""
echo "Files created:"
find /home/bolt/projects/bb/bloodBath/bloodBank/raw/pump_881235 -name "*.csv" 2>/dev/null | wc -l
echo ""
