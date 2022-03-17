#!/usr/bin/env bash

cmd=run.pl
num_extra_lm_states=2000

echo "$0 $@"  

. ./utils/parse_options.sh

if [ $# -ne 1 ]; then
  echo "Usage: $0 tree_dir"
  exit 1
fi

if [ -f path.sh ]; then
  . path.sh
fi

tree_dir=$1

echo "$0: creating denominator FST"
$cmd $tree_dir/log/make_den_fst.log \
  chain-make-den-fst $tree_dir/tree $tree_dir/0.trans_mdl $tree_dir/phone_lm.fst \
  $tree_dir/den.fst $tree_dir/normalization.fst || exit 1;
