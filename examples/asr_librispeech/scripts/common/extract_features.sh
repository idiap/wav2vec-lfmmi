#!/bin/bash

set -e

# set the wav2vec ckpt path correctly
ckpt=
stage=0
nj=100

. cmd.sh
. path.sh
. parse_options.sh

if [ $# -ne 1 ]; then
    echo "Usage: $0 data_dir"
    echo ""
    echo "This script extract features from a wav2vec model"
    exit
fi
    
data_dir=$1
[ ! -d $data_dir/feats ] && mkdir $data_dir/feats
utils/split_data.sh $data_dir $nj
if [ $stage -le 0 ]; then
    echo "$0: Extracting features..."
    for f in feats.scp cmvn.scp; do
        [ -f $data_dir/$f ] && rm $data_dir/$f
    done
    $cuda_cmd -l hostname=vgn[ef]* JOB=1:$nj $data_dir/log/extract_feats.JOB.log \
        src/examples/farsi-wav2vec2/extract_features.py \
            --wav_scp $data_dir/split${nj}/JOB/wav.scp \
            --out $data_dir/feats/feats.JOB \
            --ckpt $ckpt  || exit 1
fi

if [ $stage -le 2 ]; then
    echo "$0: Concatenating feats.scp files into one file"
    for i in `seq 1 $nj`; do
        cat $data_dir/feats/feats.$i.scp 
    done > $data_dir/feats.scp
    utils/fix_data_dir.sh $data_dir
fi

if [ $stage -le 3 ]; then
    steps/compute_cmvn_stats.sh $data_dir/ $data_dir/log $data_dir/feats
    utils/fix_data_dir.sh $data_dir
fi

