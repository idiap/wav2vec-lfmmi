#!/bin/bash
# Copyright 2020 	Srikanth Madikeri (Idiap Research Institute)

set -e

. parse_options.sh

if [ $# -ne 1 ]; then
    echo "Usage:  data_dir"
    echo "$0: This script takes a segmented data directory and creates a segments file"
    echo "based on the utt2dur file. utt2dur will be created if one doesn't exist"
    exit
fi
data_dir=$1

if [ ! -f $data_dir/utt2dur ]; then
    utils/data/get_utt2dur.sh $data_dir
fi

if [ -f $data_dir/segments ]; then
    echo "$0: $data_dir/segments already exists. Delete before rerunning the script"
else
    awk '{print $1,$1,0,$2}' $data_dir/utt2dur > $data_dir/segments
fi

if [ -f $data_dir/reco2file_and_channel  ]; then
    echo "$0: Overwriting reco2file_and_channel"
fi
awk '{print $1, $1, 1}' < $data_dir/wav.scp > $data_dir/reco2file_and_channel
