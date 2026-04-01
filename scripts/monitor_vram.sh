#!/bin/bash
# Monitor peak VRAM usage during a process
LOGFILE=$1
echo "timestamp,vram_used_mb" > $LOGFILE
while true; do
    VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    echo "$(date +%s),$VRAM" >> $LOGFILE
    sleep 1
done
