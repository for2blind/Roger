#!/bin/bash
until python3 scheduler.py; do
    echo "scheduler.py crashed with exit code $?. Respawning.." >&2
    sleep 1
done
