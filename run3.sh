#!/bin/bash
# Resume: ablate + eval from the already-computed dirs2.pt (extraction/scoring done).
set -euo pipefail
cd /home/honey/ablit
echo $$ > run3.pid
rm -f out/DONE3
exec >run3.log 2>&1

DK="docker run --rm -v /srv/training/models:/models:ro -v /home/honey/ablit:/work qwen-ablit python3 -u"
BASE=/models/MiMo-V2.6-Distill-Qwen-9B

# fresh, world-writable output dirs so the container uid can write shards
rm -rf out/C out/D
mkdir -p out/C out/D
chmod 777 out/C out/D

echo "=== [1/3] ablate C (band 10:31, clean)  $(date) ==="
$DK /work/ablate.py --dirs /work/out/dirs2.pt --out /work/out/C \
    --band 10:31 --chunks 4 --k 1 \
    --alpha-attn 1.0 --alpha-mlp 1.0 --no-biproj --no-embed

echo "=== [2/3] ablate D (band 6:31 + embed, aggressive)  $(date) ==="
$DK /work/ablate.py --dirs /work/out/dirs2.pt --out /work/out/D \
    --band 6:31 --chunks 5 --k 1 \
    --alpha-attn 1.0 --alpha-mlp 1.0 --alpha-emb 1.0 --no-biproj

echo "=== [3/3] eval base vs C vs D  $(date) ==="
$DK /work/eval.py --base "$BASE" --extra C=/work/out/C D=/work/out/D \
    --harmful /work/data/harmful.json --harmless /work/data/harmless.json \
    --n-harm 32 --n-plain 32 --new-tokens 40 --batch 16 \
    --report /work/out/report2.json

echo "=== DONE  $(date) ==="
touch out/DONE3
