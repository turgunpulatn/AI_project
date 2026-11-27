#!/usr/bin/env bash
# Retrain after the "uz/en/ru only + realistic phone-speaker / laptop-mic augmentation" change.
# Log: results/retrain_mic.log   Previous model: models_backup_before_mic/
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
LOG=results/retrain_mic.log
: > "$LOG"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
log "PREPARE start"
python -u scripts/prepare_data.py --aug 2 --max-windows 2 --boost 3 --workers 10 2>&1 \
  | grep -v -iE "warn|symlink" | grep -vE "^\s+[0-9]+/" >> "$LOG" && test -f data/processed/wavlm.npy \
  || { log "PREPARE FAILED"; exit 1; }
log "TRAIN start"
if ! python -u train.py --epochs 12 2>&1 | grep -v -iE "warn|symlink" >> "$LOG"; then
  log "TRAIN FAILED - restoring previous model"; rm -rf models && cp -r models_backup_before_mic models; exit 1
fi
log "PLAYBACK start"
python -u train_playback.py --epochs 12 --workers 10 2>&1 | grep -v -iE "warn|symlink" >> "$LOG" || log "PLAYBACK FAILED"
log "ALL DONE"
