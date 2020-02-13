#!/bin/bash

export BERT_BASE_DIR=/root/work/jawiki_sp_model/jamodel

counter=0
while read line
do
    counter=$((counter+1))
    echo "$line" > enc_inputs/${counter}.txt
done < "${1:-/dev/stdin}"

CUDA_VISIBLE_DEVICES=-1 python extract_features.py \
		    --input_file=enc_inputs \
		    --output_file=enc_outputs \
		    --vocab_file=$BERT_BASE_DIR/vocab.txt \
		    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
		    --init_checkpoint=$BERT_BASE_DIR/model/model.ckpt-1000000 \
		    --layers=-1,-2,-3,-4 \
		    --max_seq_length=128 \
		    --batch_size=8
