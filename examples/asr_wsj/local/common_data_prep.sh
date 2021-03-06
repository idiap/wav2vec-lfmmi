#!/bin/bash
# Copyright (c) Yiwen Shao, Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The common data preparation script for hybrid systems

set -euo pipefail

stage=-10
nj=30
train_set=train_si284
test_set="test_dev93 test_eval92"

wsj0=
wsj1=
if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
  wsj0=/export/corpora5/LDC/LDC93S6B
  wsj1=/export/corpora5/LDC/LDC94S13B
fi

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh


if [ $stage -le -4 ]; then
  # data preparation
  [ -z $wsj0 ] || [ -z $wsj1 ] && echo "Specify '--wsj0' and '--wsj1' as the paths to the corpus" && exit 1;
  local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?
  local/wsj_prepare_dict.sh --dict-suffix "_nosp"
  utils/prepare_lang.sh data/local/dict_nosp \
                        "<SPOKEN_NOISE>" data/local/lang_tmp_nosp data/lang_nosp
  local/wsj_format_data.sh --lang-suffix "_nosp"
  echo "Done formatting the data."

  local/wsj_extend_dict.sh --dict-suffix "_nosp" $wsj1/13-32.1
  utils/prepare_lang.sh data/local/dict_nosp_larger \
                        "<SPOKEN_NOISE>" data/local/lang_tmp_nosp_larger \
                        data/lang_nosp_bd
  local/wsj_train_lms.sh --dict-suffix "_nosp"
  local/wsj_format_local_lms.sh --lang-suffix "_nosp"
  echo "Done exteding the dictionary and formatting LMs."
fi

if [ $stage -le -3 ]; then
  # make MFCC features for the test data
  echo "$0: extracting MFCC features for the test sets"
  for dataset in $test_set; do
    mv data/$dataset data/${dataset}_hires
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj \
      --mfcc-config conf/mfcc_hires.conf data/${dataset}_hires
    steps/compute_cmvn_stats.sh data/${dataset}_hires
  done
fi

if [ $stage -le -2 ]; then
  echo "$0: perturbing the training data"
  utils/data/get_utt2dur.sh data/$train_set
  utils/data/perturb_data_dir_speed_3way.sh data/${train_set} data/${train_set}_sp
  utils/copy_data_dir.sh data/${train_set}_sp data/${train_set}_sp_hires

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_hires
fi

if [ $stage -le -1 ]; then
  echo "$0: extracting MFCC features for the training data"
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj \
    --mfcc-config conf/mfcc_hires.conf data/${train_set}_sp_hires
  steps/compute_cmvn_stats.sh data/${train_set}_sp_hires
  utils/fix_data_dir.sh data/${train_set}_sp_hires
fi

exit 0;
