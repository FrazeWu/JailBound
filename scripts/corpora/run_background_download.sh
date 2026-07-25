#!/bin/bash
# Run dataset download in background with timeout handling

export CI="true"
export DEBIAN_FRONTEND="noninteractive"
export GIT_TERMINAL_PROMPT="0"
export GCM_INTERACTIVE="never"

echo "Starting dataset download..."
echo "Log will be saved to: downloaded_datasets/download.log"

python3 download_datasets.py &
PID=$!

# Wait for process or timeout after 2 hours
for i in {1..7200}; do
    if ! kill -0 $PID 2>/dev/null; then
        echo "Download process completed!"
        exit 0
    fi
    sleep 1
done

echo "Timeout reached, killing download process..."
kill $PID 2>/dev/null
exit 1
