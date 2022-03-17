#!/bin/bash

on_error(){
  >&2 echo "Error: $1"
  exit 2
}

root="$PWD"
#
condapath=`which conda`
get_free_port="${root}/scripts/get_free_port.py"

gpu_queue="sgpu"
# model and data related
tree_dir=exp/chain/e2e_tree
manifest=data/chain_e2e
dir=''
config_file=
num_targets=

w2v_path=
max_tokens=800000
update_freq=2
n_nodes=2
n_updates=10000
mask_updates=8000
freeze_updates=500
tdnnf_grad_mult=20
updates_per_job=5000
n_updates_start=20000
reset_dl=False
reset_op=False

. ./utils/parse_options.sh

echo $manifest 
echo $tree_dir
echo $gpu_queue

warmup_updates=$((n_updates * 1 / 10))
hold_updates=$((n_updates * 4 / 10))
decay_updates=$((n_updates * 5 / 10))

echo $num_targets
echo $reset_op
echo $reset_dl

logroot="${dir}/log_train/"
mkdir -p logs/train
mkdir -p $logroot
mkdir -p "${root}/jobs/master_info"


echo $n_updates_start
echo $n_updates
echo $updates_per_job
echo $w2v_path
echo $manifest
echo $config_file
echo $warmup_updates

for (( target_updates=${n_updates_start}; target_updates<=${n_updates}; target_updates+=${updates_per_job}))
do
  echo $target_updates
  #################### Prepare Commands ####################
  # --finetune-from-model=${root}/models/wav2vec_small.pt
  IFS= read -d '' cmd <<EOF
${condapath}/bin/fairseq-hydra-train
task.data=${root}/${manifest}
task.num_targets=${num_targets}
dataset.max_tokens=${max_tokens}
checkpoint.save_dir=${root}/${dir}
checkpoint.reset_dataloader=${reset_dl}
checkpoint.reset_optimizer=${reset_op}
criterion.denominator_fst_path=${root}/$tree_dir/den.fst
optimization.update_freq=[$update_freq]
optimization.max_update=${target_updates}
lr_scheduler.warmup_steps=${warmup_updates}
lr_scheduler.hold_steps=${hold_updates}
lr_scheduler.decay_steps=${decay_updates}
model.mask_updates=$mask_updates
model.freeze_finetune_updates=$freeze_updates
model.tdnnf_grad_mult=$tdnnf_grad_mult
model.w2v_path=$w2v_path
--config-dir=${root}/config
--config-name=${config_file}
EOF
  cmd=$(echo "${cmd}" | tr '\n' ' ')
  #################### Prepare Commands (end) ##############
  
  >&2 echo "==================================================" &&
  >&2 echo "Submit jobs for target : ${target_updates} updates" &&
  info_file=$(mktemp -p "${root}/jobs/master_info") &&
  >&2 echo "Submit distributed training jobs" &&
  >&2 echo "Master node info file: ${info_file}" &&
  >&2 echo "Command:" &&
  >&2 echo "  ${cmd}" &&
  [ -z "${depend_jids}" ] &&
    depend_opt='' ||
    depend_opt="-hold_jid ${depend_jids}"
  depend_jids='' &&
  for (( node_id=0; node_id<${n_nodes}; node_id++))
  do
    >&2 echo "target:${target_updates}; node:${node_id}"
    echo ${node_id}
    # The following need to be updated to the job submission protocol at your institute
    jid=$(
      qsub \
        -terse \
        ${depend_opt} \
        -N espresso_sgpu_train \
        -S /bin/bash -cwd -V \
        -o 'logs/train/$JOB_NAME_$JOB_ID.stdout' \
        -e 'logs/train/$JOB_NAME_$JOB_ID.stderr' \
<<EOF
source ${HOME}/.bashrc
(
  echo "==================================================" &&
  echo "Job for target ${target_updates} updates started at \$(date)" &&
  >&2 echo "==================================================" &&
  >&2 echo "Job for target ${target_updates} updates started at \$(date)" &&
  >&2 echo "node ($node_id) @ \$(hostname)" &&
  if [ "${node_id}" -eq 0 ]
  then
    master_host=\$(hostname) &&
    master_port=\$(${get_free_port})
    echo "\$master_host \$master_port" > ${info_file}
  else
    while [ "\$(wc -l ${info_file} | awk '{print \$1}')" -lt 1 ]
    do
      >&2 echo "wait for master info..."
      sleep 1
    done
    read master_host master_port < ${info_file}
  fi
  >&2 echo "master at \${master_host}:\${master_port}" &&
  >&2 echo "${cmd}" &&
  time python -m torch.distributed.launch --use_env --nproc_per_node=1 \
    --nnodes=${n_nodes} --node_rank=${node_id} \
    --master_addr="\${master_host}" --master_port="\${master_port}" \
    ${cmd}
  echo "Job for node ($node_id) @ \$(hostname) finished at \$(date)" >&2
) > >(tee -a '${logroot}/stdout.log.${node_id}') 2> >(tee -a '${logroot}/stderr.log.${node_id}' >&2)
EOF
    ) &&
    >&2 echo "Submitted job #${jid}" &&
    depend_jids="${depend_jids:+${depend_jids},}${jid}"
  done
done
