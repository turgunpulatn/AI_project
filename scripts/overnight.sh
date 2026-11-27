#!/usr/bin/env bash
# Overnight pipeline: wait for data generation -> prepare -> train main model -> train playback detector.
# Log: results/overnight.log
set -o pipefail
cd "$(dirname "$0")/.."
LOG=results/overnight.log
mkdir -p results
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "waiting for generate_ai.py / make_fakes.py to finish"
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'generate_ai.py|make_fakes.py|import_fleurs.py' }) { exit 0 } else { exit 1 }"; do
  sleep 60
done
log "generation finished: human=$(ls dataset/human | wc -l) ai=$(ls dataset/ai | wc -l)"

log "PREPARE start"
if python -u scripts/prepare_data.py --aug 1 --max-windows 2 --boost 3 --workers 10 2>&1 \
     | grep -v -iE "warn|symlink" | grep -vE "^\s+[0-9]+/" >> "$LOG" && test -f data/processed/wavlm.npy; then
  log "PREPARE ok"
else
  log "PREPARE FAILED - stopping"; exit 1
fi

cp -r models "models_backup_$(date +%H%M)"   # keep the previous working model just in case
log "TRAIN start"
if python -u train.py --epochs 12 --focus-weight 2 2>&1 | grep -v -iE "warn|symlink" >> "$LOG"; then
  log "TRAIN ok"
else
  log "TRAIN FAILED - restoring previous model"
  rm -rf models && cp -r "$(ls -d models_backup_* | tail -1)" models
fi

log "PLAYBACK start"
if python -u train_playback.py --epochs 12 --workers 10 2>&1 | grep -v -iE "warn|symlink" >> "$LOG"; then
  log "PLAYBACK ok"
else
  log "PLAYBACK FAILED"
fi
log "ALL DONE"
