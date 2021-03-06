#!/bin/bash

set -e

cmd=run.pl
num_extra_lm_states=2000

echo "$0 $@"

. ./utils/parse_options.sh

echo $#
if [ $# -ne 7 ]; then
  echo "Usage: $0 lang_dir tree_dir data_dir shared_phone phones_type njobs"
  echo "Example usage: shutil/chain/estimate_e2e_phone_lm.sh data/lang_lp exp/chain/e2e_tree data/train_spe2e_hires true biphone lang_e2e_biphone 32"
  exit 1
fi

if [ -f path.sh ]; then
  . path.sh
fi

langdir=$1
treedir=$2
trainset=$3
shared_phones=$4
phones_type=$5
newlangdir=$6
nj=$7

echo "$0: Estimating a phone language model for the denominator graph..."
mkdir -p $treedir/log
run.pl $treedir/log/make_phone_lm.log \
  cat $trainset/text \| \
  steps/nnet3/chain/e2e/text_to_phones.py --between-silprob 0.1 \
  $langdir \| \
  utils/sym2int.pl -f 2- $langdir/phones.txt \| \
  chain-est-phone-lm --num-extra-lm-states=2000 \
  ark:- $treedir/phone_lm.fst

steps/nnet3/chain/e2e/prepare_e2e.sh --nj $nj --cmd "$cmd" \
  --shared-phones $shared_phones \
  --type $phones_type \
  $trainset $newlangdir $treedir

