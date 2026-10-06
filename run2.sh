#!/bin/bash
# Full corrected abliteration pipeline for MiMo-V2.6-Distill-Qwen-9B.
# Response-region extraction -> direction scoring -> MPOA ablation (2 variants)
# -> refusal + capability eval. Base model is read-only; all writes go to out/.
set -euo pipefail
cd /home/honey/ablit
echo $$ > run2b.pid
rm -f out/DONE2
exec >run2b.log 2>&1

DK="docker run --rm -v /srv/training/models:/models:ro -v /home/honey/ablit:/work qwen-ablit python3 -u"
BASE=/models/MiMo-V2.6-Distill-Qwen-9B
mkdir -p out/C out/D

echo "=== [1/5] extract (response region)  $(date) ==="
$DK /work/extract2.py --harmful /work/data/harmful.json --harmless /work/data/harmless.json \
    --n 64 --batch 16 --resp 8 --max-len 64 --out /work/out/acts2.pt

echo "=== [2/5] score directions  $(date) ==="
$DK /work/score.py --acts /work/out/acts2.pt --out /work/out/dirs2.pt

echo "=== [3/5] ablate C (band 10:31, clean)  $(date) ==="
$DK /work/ablate.py --dirs /work/out/dirs2.pt --out /work/out/C \
    --band 10:31 --chunks 4 --k 1 \
    --alpha-attn 1.0 --alpha-mlp 1.0 --no-biproj --no-embed

echo "=== [4/5] ablate D (band 6:31 + embed, aggressive)  $(date) ==="
$DK /work/ablate.py --dirs /work/out/dirs2.pt --out /work/out/D \
    --band 6:31 --chunks 5 --k 1 \
    --alpha-attn 1.0 --alpha-mlp 1.0 --alpha-emb 1.0 --no-biproj

echo "=== [5/5] eval base vs C vs D  $(date) ==="
$DK /work/eval.py --base "$BASE" --extra C=/work/out/C D=/work/out/D \
    --harmful /work/data/harmful.json --harmless /work/data/harmless.json \
    --n-harm 32 --n-plain 32 --new-tokens 40 --batch 16 \
    --report /work/out/report2.json

echo "=== DONE  $(date) ==="
touch out/DONE2
