#!/bin/bash
echo "Running with SPEAKER=${SPEAKER}, PACE=${PACE}, PITCH_VARIATION=${PITCH_VARIATION}, DENOISE=${DENOISE}"

python /app/process_srt.py \
    --speaker "${SPEAKER:-1}" \
    --pace "${PACE:-0.9}" \
    --pitch-variation "${PITCH_VARIATION:-0.1}" \
    --denoise "${DENOISE:-0.005}"
