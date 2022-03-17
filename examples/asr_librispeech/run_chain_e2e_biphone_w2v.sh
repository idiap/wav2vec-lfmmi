#!/bin/bash
# Copyright (c) Yiming Wang, Yiwen Shao
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e -o pipefail

stage=-10
stop_stage=9999
ngpus=1 # num GPUs for multiple GPUs training within a single node; should match those in $free_gpu
free_gpu= # comma-separated available GPU ids, eg., "0" or "0,1"; automatically assigned if on CLSP grid
gpu_queue='sgpu'

# model training related
batch_size=16
update_freq=8
n_nodes=4
config_file=w2v_tdnnf_clean
max_tokens=800000
max_update=10000
mask_updates=8000
freeze_updates=500
tdnnf_grad_mult=20
# Set the wav2vec path correctly
w2v_path=

reset_op=False 
reset_dl=False

# Training loop related options
# Set them to much lower numbers if using short_gpu (not recommended)
updates_per_job=7000
n_updates_start=7000

# Update the corpus paths to your own
corpus=./database/LibriSpeech
lm_url=www.openslr.org/resources/11

# model and data related
affix=
lang=data/lang_chain_e2e
tree_dir=exp/chain/e2e_tree  # it's actually just a trivial tree (no tree building)
shared_phones=true
phones_type=biphone

unperturbed_train_set=train_clean_100
whole_train_set=${unperturbed_train_set}_sp  # will be split into train_set and valid_set
train_set=${unperturbed_train_set}_novalid_spe2e
valid_set=${unperturbed_train_set}_valid_spe2e
test_set="test_clean dev_clean test_other dev_other"

dumpdir=data/dump   # directory to dump full features
testitr='test'
phones_type=biphone

manifest=data/chain_e2e_wavs

if [ -f path.sh ]; then . ./path.sh; fi
. ./cmd.sh
. ./utils/parse_options.sh

#dir=exp/chain/e2e${affix:+_$affix}
dir=exp/chain/e2e${affix:+_$affix}

echo $dir
if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  local/download_lm.sh $lm_url data/local/lm
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  # format the data as Kaldi data directories
  echo "Stage 0: Formatting data directories"
  if [ ! -d "data/local/lm" ]; then
      echo "Exiting because data/local/lm folder doesn't exist"
      echo "Please set it up by either of two options:"
      echo "running from stage -1 OR copying/linking to existing folders"
      exit
  fi
  for part in test-clean dev-clean test-other dev-other train-clean-100 train-clean-360 train-other-500; do
    # use underscore-separated names in data directories.
    local/data_prep.sh $corpus/$part data/$(echo $part | sed s/-/_/g)
  done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "Stage 1: Setting Language Model"
  mkdir -p data/local/lm_less_phones
  ln -rs data/local/lm/3-gram.arpa.gz data/local/lm_less_phones/lm_tglarge.arpa.gz
  ln -rs data/local/lm/3-gram.pruned.1e-7.arpa.gz data/local/lm_less_phones/lm_tgmed.arpa.gz
  ln -rs data/local/lm/3-gram.pruned.3e-7.arpa.gz data/local/lm_less_phones/lm_tgsmall.arpa.gz
  ln -rs data/local/lm/4-gram.arpa.gz data/local/lm_less_phones/lm_fglarge.arpa.gz
  cp data/local/lm/librispeech-vocab.txt data/local/lm_less_phones/
  cat data/local/lm/librispeech-lexicon.txt | sed -e 's/[0,1,2]//g' > \
    data/local/lm_less_phones/librispeech-lexicon.txt

  local/prepare_dict.sh --stage 3 --nj 30 --cmd "$cpu_cmd" \
    data/local/lm_less_phones data/local/lm_less_phones data/local/dict_lp

  utils/prepare_lang.sh \
    --position_dependent_phones false \
    --share_silence_phones true \
    data/local/dict_lp \
    "<UNK>" data/local/lang_tmp_lp data/lang_lp

  local/format_lms.sh --src-dir data/lang_lp data/local/lm_less_phones
  # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
  utils/build_const_arpa_lm.sh data/local/lm_less_phones/lm_tglarge.arpa.gz \
    data/lang_lp data/lang_lp_test_tglarge
  utils/build_const_arpa_lm.sh data/local/lm_less_phones/lm_fglarge.arpa.gz \
    data/lang_lp data/lang_lp_test_fglarge
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then

  echo "Stage 2: Extracting Features"
  for part in ${test_set}; do
    utils/copy_data_dir.sh data/$part data/${part}_hires
    utils/data/get_utt2dur.sh data/${part}_hires # necessary for the next command
  done

  # utils/combine_data.sh data/train_960/ \
  #   data/train_clean_100 \
  #   data/train_clean_360 \
  #   data/train_other_500

  echo "$0: perturbing the training data to allowed lengths"
  # Uncomment the following command block if you need to firt exptract segments
  # For example in switchboard data preparation

  # utils/data/extract_wav_segments_data_dir.sh --nj 66 \
  #   --cmd "$cpu_cmd" data/$trainset data/${trainset}_wav
  # utils/data/get_utt2dur.sh --nj 66 --cmd "$cpu_cmd" \
  #   data/${trainset}_wav # necessary for the next command
  # # 12 in the following command means the allowed lengths are spaced
  # # by 12% change in length.
  # utils/data/perturb_speed_to_allowed_lengths.py 12 \
  #   data/${trainset}_wav \
  #   data/${trainset}_sp_hires

  # Comment out the following two commands if you uncomment the above block 
  utils/data/get_utt2dur.sh data/${unperturbed_train_set} # necessary for the next command
  # 12 in the following command means the allowed lengths are spaced
  # by 12% change in length.
  utils/data/perturb_data_dir_speed_3way.sh data/${unperturbed_train_set} \
    data/${unperturbed_train_set}_sp
  utils/copy_data_dir.sh data/${unperturbed_train_set}_sp data/${unperturbed_train_set}_sp_hires

  ## cat data/${unperturbed_train_set}_sp_hires/utt2dur | \
  ##   awk '{print $1 " " substr($1,5)}' >data/${unperturbed_train_set}_sp_hires/utt2uniq
  utils/data/perturb_data_dir_volume.sh data/${unperturbed_train_set}_sp_hires
  utils/fix_data_dir.sh data/${unperturbed_train_set}_sp_hires

  # for part in ${test_set} ${whole_train_set}; do
  #   datadir=${part}_hires
  #   # Extracting 80 dim filter bank features
  #   mkdir -p data/feats/fbank
  #   steps/make_fbank.sh --fbank-config conf/fbank_hires.conf \
  #     --cmd "$cpu_cmd" --nj 50 data/${datadir} \
  #     data/feats/fbank/${datadir} data/feats/fbank/${datadir}/data || exit 1;
  #   steps/compute_cmvn_stats.sh data/${datadir} \
  #     data/feats/fbank/${datadir} data/feats/fbank/${datadir}/data || exit 1;
  #   utils/fix_data_dir.sh data/${datadir} || exit 1
  # done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "Stage 3: Create the $lang Directory that Has a Specific HMM Topolopy"
  rm -rf $lang
  bash shutil/chain/check_lang.sh data/lang_lp $lang
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo "Stage 4: Generate Denominator Graph and Numerator Fsts"
  echo "$0: Estimating a phone language model for the denominator graph..."
  mkdir -p $tree_dir/log

  nj=32
  bash shutil/chain/estimate_e2e_phone_lm.sh --cmd "$train_cmd" \
    data/lang_lp $tree_dir \
    data/${whole_train_set}_hires \
    $shared_phones $phones_type $lang $nj

  # echo "$0: Making denominator fst..."
  bash shutil/chain/make_e2e_den_fst.sh --cmd "$decode_cmd" \
    $tree_dir || exit 1
  echo "$0: Making numerator fsts..."
  abs_treedir=`utils/make_absolute.sh $tree_dir`
  $decode_cmd JOB=1:$nj $tree_dir/log/make_num_fst_e2e.JOB.log \
    chain-make-num-fst-e2e $tree_dir/0.trans_mdl $tree_dir/normalization.fst \
      scp:$tree_dir/fst.JOB.scp \
      ark,scp:$abs_treedir/fst_nor.JOB.ark,$abs_treedir/fst_nor.JOB.scp || exit 1
  for n in $(seq $nj); do
    cat $tree_dir/fst_nor.$n.scp || exit 1
  done > $tree_dir/fst_nor.scp || exit 1
fi

if [ ${stage} -le 5 ] && [ $stop_stage -ge 5 ]; then
  echo "Stage 5: Split the Whole Train Set into Train/Valid Set"
  # Get list of validation utterances.
  data=data/${whole_train_set}_hires
  set +e
  awk '{print $1}' $data/utt2spk | utils/shuffle_list.pl 2>/dev/null | head -300 > valid_uttlist
  set -e
  if [ -f $data/utt2uniq ]; then  # this matters if you use data augmentation.
    echo "File $data/utt2uniq exists, so augmenting valid_uttlist to"
    echo "include all perturbed versions of the same 'real' utterances."
    mv valid_uttlist valid_uttlist.tmp
    utils/utt2spk_to_spk2utt.pl $data/utt2uniq > uniq2utt
    cat valid_uttlist.tmp | utils/apply_map.pl $data/utt2uniq | \
      sort | uniq | utils/apply_map.pl uniq2utt | \
      awk '{for(n=1;n<=NF;n++) print $n;}' | sort  > valid_uttlist
    rm uniq2utt valid_uttlist.tmp 2>/dev/null
  fi
  # generate train/valid data dir
  utils/filter_scp.pl --exclude valid_uttlist $data/utt2spk | cut -d" " -f1 > novalid_uttlist || exit 1
  utils/subset_data_dir.sh --utt-list novalid_uttlist $data data/${train_set}_hires || exit 1
  utils/subset_data_dir.sh --utt-list valid_uttlist $data data/${valid_set}_hires || exit 1

  # generate train/valid numerator fst file
  utils/filter_scp.pl novalid_uttlist $tree_dir/fst_nor.scp > $tree_dir/fst_novalid_nor.scp || exit 1
  utils/filter_scp.pl valid_uttlist $tree_dir/fst_nor.scp > $tree_dir/fst_valid_nor.scp || exit 1
  rm valid_uttlist novalid_uttlist 2>/dev/null

  # not all fsts can be generated successfully, just filter out those not having the fst
  for dataset in $train_set $valid_set; do
    tag=novalid && [[ "$dataset" == "$valid_set" ]] && tag=valid
    cp data/${dataset}_hires/wav.scp data/${dataset}_hires/wav.scp.tmp
    utils/filter_scp.pl $tree_dir/fst_${tag}_nor.scp data/${dataset}_hires/wav.scp.tmp \
      > data/${dataset}_hires/wav.scp || exit 1
    rm data/${dataset}_hires/wav.scp.tmp 2>/dev/null
    utils/fix_data_dir.sh data/${dataset}_hires || exit 1
  done
fi

if [ ${stage} -le 6 ] && [ $stop_stage -ge 6 ]; then
  echo "Stage 6: Dump Feature"
  for dataset in $train_set $valid_set $test_set; do
    nj=30
    utils/split_data.sh data/${dataset}_hires $nj
    sdata=data/${dataset}_hires/split$nj
    mkdir -p $dumpdir/${dataset}_hires; abs_featdir=`utils/make_absolute.sh $dumpdir/${dataset}_hires`
    utils/copy_data_dir.sh data/${dataset}_hires $abs_featdir
  done
fi

if [ ${stage} -le 7 ] && [ $stop_stage -ge 7 ]; then
  echo "Stage 7: Make Graphs"
  for lmtype in tgsmall; do
    utils/lang/check_phones_compatible.sh \
      data/lang_lp_test_${lmtype}/phones.txt $lang/phones.txt
    utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov \
      data/lang_lp_test_$lmtype $tree_dir $tree_dir/graph_$lmtype || exit 1
  done
fi

# if [ ${stage} -le 8 ] && [ $stop_stage -ge 8 ]; then
#   echo "Stage 8: Dump Json Files"
#   train_wav=$dumpdir/${train_set}_hires/wav.scp
#   train_fst=${tree_dir}/fst_novalid_nor.scp
#   train_text=data/${train_set}_hires/text
#   valid_wav=$dumpdir/${valid_set}_hires/wav.scp
#   valid_fst=${tree_dir}/fst_valid_nor.scp
#   valid_text=data/${valid_set}_hires/text
#   mkdir -p data/chain_e2e_wavs
#   asr_wav_prep_json.py --wav-files $train_wav --numerator-fst-files $train_fst --text-files $train_text \
#     --output data/chain_e2e_wavs/train.json
#   asr_wav_prep_json.py --wav-files $valid_wav --numerator-fst-files $valid_fst --text-files $valid_text \
#     --output data/chain_e2e_wavs/valid.json
#   for dataset in $test_set; do
#     nj=$(wc -l <data/${dataset}_hires/spk2utt)
#     utils/split_data.sh data/${dataset}_hires $nj
#     utils/split_data.sh $dumpdir/${dataset}_hires $nj
#     for n in $(seq $nj); do
#       wav=$dumpdir/${dataset}_hires/split$nj/$n/wav.scp
#       text=data/${dataset}_hires/split$nj/$n/text
#       asr_wav_prep_json.py --wav-files $wav --text-files $text \
#         --output data/chain_e2e_wavs/$dataset.$n.json
#     done
#   done
# fi


if [ ${stage} -le 8 ] && [ $stop_stage -ge 8 ]; then
  echo "Stage 9: Dump Json Files"
  train_wav=$dumpdir/${train_set}_hires/wav.scp
  train_fst=${tree_dir}/fst_novalid_nor.scp
  train_text=data/${train_set}_hires/text
  valid_wav=$dumpdir/${valid_set}_hires/wav.scp
  valid_fst=${tree_dir}/fst_valid_nor.scp
  valid_text=data/${valid_set}_hires/text
  valid_dur=data/${valid_set}_hires/utt2dur


  mkdir -p $manifest
  tmp_dir=$(mktemp -d --tmpdir=$manifest)
  nj_dump=40
  utils/split_data.sh $dumpdir/${train_set}_hires $nj_dump

  $decode_cmd JOB=1:$nj_dump $manifest/logs/dump_train_json.JOB.log \
    asr_wav_prep_json_v2.py \
      --wav-files $dumpdir/${train_set}_hires/split${nj_dump}/JOB/wav.scp \
      --numerator-fst-files $train_fst \
      --text-files $dumpdir/${train_set}_hires/split${nj_dump}/JOB/text \
      --duration-files $dumpdir/${train_set}_hires/split${nj_dump}/JOB/utt2dur \
      --sampling-rate 16000 \
      --output ${tmp_dir}/train.JOB.json


  # Filter files with max lengths 30 seconds and which have corrupt numerator fsts
  $decode_cmd JOB=1:$nj_dump $manifest/logs/dump_train_filter_json.JOB.log \
    asr_filter_wav_json.py \
      --json-file ${tmp_dir}/train.JOB.json \
      --output ${tmp_dir}/train.filtered.JOB.json \
      --max-length 480000

  train_json_files=`ls ${tmp_dir}/train.filtered.*.json`
  asr_merge_wav_json.py --json-files $train_json_files --output $manifest/train.json
  rm $tmp_dir/*
  rmdir $tmp_dir

  asr_wav_prep_json_v2.py \
    --wav-files $valid_wav \
    --numerator-fst-files $valid_fst \
    --text-files $valid_text \
    --duration-files $valid_dur \
    --sampling-rate 16000 \
    --output $manifest/valid.json

  for dataset in $test_set; do
    nspk=$(wc -l <data/${dataset}_hires/spk2utt)
    nj_max=40
    nj=$(( nspk > nj_max ? nj_max : nspk ))

    utils/data/get_utt2dur.sh data/${dataset}_hires # necessary for the next command
    utils/data/get_utt2dur.sh $dumpdir/${dataset}_hires # necessary for the next command

    rm -rf data/${dataset}_hires/split$nj
    rm -rf $dumpdir/${dataset}_hires/split$nj

    utils/split_data.sh data/${dataset}_hires $nj
    utils/split_data.sh $dumpdir/${dataset}_hires $nj

    for n in $(seq $nj); do
      wav=$dumpdir/${dataset}_hires/split$nj/$n/wav.scp
      text=data/${dataset}_hires/split$nj/$n/text
      dur=data/${dataset}_hires/split$nj/$n/utt2dur
      asr_wav_prep_json_v2.py \
        --wav-files $wav \
        --text-files $text \
        --duration-files $dur \
        --sampling-rate 16000 \
        --output $manifest/$dataset.$n.json
    done
  done
  #mv $manifest/train.json $manifest/train.json.full
fi

if [ ${stage} -le 9 ] && [ $stop_stage -ge 9 ]; then
  echo "Stage 10: Filter training utterances"
  # asr_filter_wav_json.py \
  #   --json-file $manifest/train.json.full \
  #   --output $manifest/train.json \
  #   --max-length 480000

  asr_fix_wav_paths.py \
    --json-file $manifest/train.json \
    --output $manifest/train.json

  for dataset in $test_set; do
    nspk=$(wc -l <data/${dataset}_hires/spk2utt)
    nj_max=40
    nj=$(( nspk > nj_max ? nj_max : nspk ))
    for n in $(seq $nj); do
      asr_fix_wav_paths.py \
        --json-file $manifest/$dataset.$n.json \
        --output $manifest/$dataset.$n.json
    done
  done
fi

if [ ${stage} -le 10 ] && [ $stop_stage -ge 10 ]; then
  # This stage has been designed with the job scheduler at the Idiap Research
  # Institute This needs to be modified accordingly for distributed training
  # scheduler.  The most important command that is run on multiple gpus can be
  # found between lines: 56-74 in submit_hydra_looped.sh
  num_targets=$(tree-info ${tree_dir}/tree | grep num-pdfs | awk '{print $2}')
  echo "Stage 11: Model Training"
  valid_subset=valid
  mkdir -p $dir/log
  log_file=$dir/log/train.log
  [ -f $dir/checkpoint_last.pt ] && log_file="-a $log_file"
  echo "Submitting Jobs"
  echo "nupdates_start $n_updates_start"
  echo "nupdates_pj $updates_per_job"
  echo "w2v-path  $w2v_path"
  bash submit_hydra_looped.sh \
    --gpu-queue $gpu_queue \
    --tree-dir $tree_dir \
    --manifest $manifest \
    --dir $dir \
    --config-file $config_file \
    --num-targets $num_targets \
    --w2v-path $w2v_path \
    --max-tokens $max_tokens \
    --update-freq $update_freq \
    --n-nodes $n_nodes \
    --n-updates $max_update \
    --mask-updates $mask_updates \
    --freeze-updates $freeze_updates \
    --tdnnf-grad-mult $tdnnf_grad_mult \
    --reset-op $reset_op \
    --reset-dl $reset_dl \
    --updates-per-job $updates_per_job \
    --n-updates-start $n_updates_start
fi

if [ ${stage} -le 11 ] && [ $stop_stage -ge 11 ]; then
  num_targets=$(tree-info ${tree_dir}/tree | grep num-pdfs | awk '{print $2}')
  echo "Stage 10: Decoding"
  rm $dir/.error 2>/dev/null || true
  queue_opt=""
  path=$dir/checkpoint_$testitr.pt
  for dataset in $test_set; do
    (
      data_affix=$(echo $dataset | sed s/test_//)
      # CAUTION CHANGING nj requires redumping features
      nj=$(wc -l <data/${dataset}_hires/spk2utt)
      # nspk=$(wc -l <data/${dataset}_hires/spk2utt)
      # nj_max=40
      # nj=$(( nspk > nj_max ? nj_max : nspk ))
      for lmtype in tgsmall; do
        graph_dir=$tree_dir/graph_${lmtype}
        $decode_cmd $queue_opt JOB=1:$nj $dir/$testitr/decode_${lmtype}_${data_affix}/log/decode.JOB.log \
          dump_posteriors.py \
            data/chain_e2e_wavs \
            --cpu \
            --task asr_wav_finetune_hybrid \
            --max-tokens 1280000 \
            --batch-size 16 \
            --num-shards 1 \
            --shard-id 0 \
            --num-targets $num_targets \
            --gen-subset $dataset.JOB \
            --path $path \| \
          latgen-faster-mapped --max-active=7000 --min-active=20 --beam=15 --lattice-beam=8 --acoustic-scale=1.0 \
            --allow-partial=true --word-symbol-table="$graph_dir/words.txt" \
            $tree_dir/0.trans_mdl $graph_dir/HCLG.fst ark:- \
            "ark:| lattice-scale --acoustic-scale=10.0 ark:- ark:- | gzip -c >$dir/$testitr/decode_${lmtype}_${data_affix}/lat.JOB.gz" || exit 1

        if [ ! -f  $dir/$testitr/decode_${lmtype}_${data_affix}/../final.mdl ]; then
          ln -s $PWD/$tree_dir/0.trans_mdl $dir/$testitr/decode_${lmtype}_${data_affix}/../final.mdl
        fi
        local/score.sh --cmd "$decode_cmd" data/${dataset}_hires $graph_dir $dir/$testitr/decode_${lmtype}_${data_affix} || exit 1
        echo $nj > $dir/$testitr/decode_${lmtype}_${data_affix}/num_jobs
      done
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_lp_test_{tgsmall,fglarge} \
        data/${dataset}_hires $dir/$testitr/decode_tgsmall_${data_affix}{,_fg} || exit 1
  ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
  for dataset in $test_set; do
    data_affix=$(echo $dataset | sed s/test_//)
    for x in $dir/${testitr}/decode_{tgsmall_${data_affix},tgsmall_${data_affix}_fg}; do
      grep WER $x/wer_* | utils/best_wer.sh
    done
  done
fi
