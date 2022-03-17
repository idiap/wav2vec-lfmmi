# you can change cmd.sh depending on what type of queue you are using.
# If you have no queueing system and want to run on a local machine, you
# can change all instances 'queue.pl' to run.pl (but be careful and run
# commands one by one: most recipes will exhaust the memory on your
# machine).  queue.pl works with GridEngine (qsub).  slurm.pl works
# with slurm.  Different queues are configured differently, with different
# queue names and different ways of specifying things like memory;
# to account for these differences you can create and edit the file
# conf/queue.conf to match your queue's configuration.  Search for
# conf/queue.conf in http://kaldi-asr.org/doc/queue.html for more information,
# or search for the string 'default_config' in utils/queue.pl or utils/slurm.pl.

#export train_cmd="run.pl --mem 10G"
#export cuda_cmd="run.pl --mem 10G --gpu 1"
#export decode_cmd="run.pl --mem 4G"

# JHU setup (copy queue-freegpu.pl from ESPnet into utils/)
export train_cmd="queue.pl --mem 10G"
export cuda_cmd="queue-freegpu.pl --mem 10G --gpu 1 --config conf/gpu.conf"
export decode_cmd="queue.pl --mem 4G"