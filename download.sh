#!/bin/bash
set -e

BASE_URL="https://huggingface.co/datasets/AgMMU/AgMMU_v1/resolve/main"
DATA_DIR="./data"

usage() {
    echo "Usage: $0 [--eval-only | --full]"
    echo "  --eval-only  Download evaluation set only (~17.8 GB)"
    echo "  --full       Download everything including fine-tuning data (~550 GB)"
    echo "  (default: --eval-only)"
    exit 1
}

MODE="eval-only"
if [[ "$1" == "--full" ]]; then
    MODE="full"
elif [[ "$1" == "--eval-only" ]] || [[ -z "$1" ]]; then
    MODE="eval-only"
elif [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    usage
else
    usage
fi

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== Downloading evaluation set ==="
wget -c "${BASE_URL}/agmmu_e_filtered_hf1.json" -O agmmu_eval.json
wget -c "${BASE_URL}/images.tar.gz" -O images.tar.gz

echo "=== Extracting evaluation images ==="
tar -xzf images.tar.gz

if [[ "$MODE" == "full" ]]; then
    echo "=== Downloading fine-tuning set (~550 GB) ==="
    wget -c "${BASE_URL}/agmmu_ft_hf1.json" -O agmmu_ft.json

    for part in aa ab ac ad ae af ag ah ai aj ak; do
        wget -c "${BASE_URL}/images_ft.tar.gz.part-${part}" -O "images_ft.tar.gz.part-${part}"
    done

    echo "=== Reassembling and extracting fine-tuning images ==="
    cat images_ft.tar.gz.part-* > images_ft.tar.gz
    tar -xzf images_ft.tar.gz
    rm images_ft.tar.gz.part-*
fi

echo "=== Done ==="
echo "Data downloaded to: $DATA_DIR"
