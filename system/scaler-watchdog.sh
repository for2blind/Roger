#!/bin/bash
until python3 scaler.py; do
    echo "scaler.py crashed with exit code $?. Respawning.." >&2
    sleep 1
done
