set -x
HOSTFILE=/etc/mpi/hostfile
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset /fl-ift/med/hujunchao/git_root/OpenRLHF/data/yingxiang_baogao/end2end_template_abnormal_to_report.xlsx \
   --input_key input \
   --output_key output \
   --train_batch_size 16 \
   --micro_train_batch_size 1 \
   --max_samples_train 20000 \
   --max_samples_eval 0 \
   --pretrain /fl-ift/med/common/Qwen1.5-14B \
   --save_path ./checkpoint/qwen1.5-14b-base-sft \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 5 \
   --bf16 \
   --flash_attn \
   --learning_rate 2e-5 \
   --gradient_checkpointing \
   --load_ds_method custom \
   --l2 0.1 \
   --model_type qwen2
EOF
    # --wandb [WANDB_TOKENS]
    # --adam_offload

if [[ ${1} != "slurm" ]]; then
    export PATH=$HOME/.local/bin/:$PATH
    deepspeed --hostfile $HOSTFILE --master_port $MASTER_PORT --module $training_commands
    # deepspeed --module $training_commands

fi