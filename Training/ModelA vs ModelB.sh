#!/bin/bash

set -e
WORKDIR="/workspace/final_genome_training"
SOURCE_SCRIPT="train_compliance_final.py"

if [ ! -f "$SOURCE_SCRIPT" ]; then
    echo "ERROR: Could not find $SOURCE_SCRIPT in the current directory."
    echo "Please save the python code to $SOURCE_SCRIPT first."
    exit 1
fi

mkdir -p "$WORKDIR"
cp "$SOURCE_SCRIPT" "$WORKDIR/"
cd "$WORKDIR"

CONDA_BASE=$(conda info --base)
PYTHON_EXEC="$CONDA_BASE/envs/enformer_stable/bin/python"

echo ">>> Checking Model Weights..."
mkdir -p enformer_weights
if [ ! -f "enformer_weights/config.json" ]; then
    echo "   Downloading config.json..."
    wget -q -nc -O enformer_weights/config.json https://huggingface.co/EleutherAI/enformer-official-rough/resolve/main/config.json
fi
if [ ! -f "enformer_weights/pytorch_model.bin" ]; then
    echo "   Downloading pytorch_model.bin (1.2GB)..."
    wget -q -nc -O enformer_weights/pytorch_model.bin https://huggingface.co/EleutherAI/enformer-official-rough/resolve/main/pytorch_model.bin
fi

if [ ! -f "gencode.v44.annotation.gtf" ]; then
    echo ">>> Fetching Whole Genome GTF..."
    wget -q -nc https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz
    gunzip -f gencode.v44.annotation.gtf.gz
fi

echo ">>> Launching FINAL MASTER TRAINING..."
nohup "$PYTHON_EXEC" -u train_compliance_final.py > compliance_results.log 2>&1 &

echo ">>> RUNNING. Track progress:"
echo "    tail -f $WORKDIR/compliance_results.log"
