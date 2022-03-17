#!/bin/bash
# Copyright 2020 	Srikanth Madikeri (Idiap Research Institute)

set -e

stage=0
nj=40

. cmd.sh
#. path.sh
. parse_options.sh

if [ $# -ne 2 ]; then
    echo "Usage: $0 old_data_dir new_data_dir"
    echo ""
    echo "This script takes a data directory and prepares wav.scp from the segments file."
    echo "It is useful to prepare data folders for e2e lfmmi scripts"
    exit
fi


old_data_dir=$1
new_data_dir=$2

echo "$0: Running utt2dur"
utils/data/get_utt2dur.sh $old_data_dir

echo "$0: Splitting data"
utils/split_data.sh $old_data_dir $nj

echo "$0: Segmenting wav files"
utils/data/extract_wav_segments_data_dir.sh \
  --cmd "$cpu_cmd" \
  --nj $nj \
  $old_data_dir \
  $new_data_dir
