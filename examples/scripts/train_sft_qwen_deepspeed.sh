set -x
# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/fl-ift/med/hujunchao/git_root/OpenRLHF
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATASET='data/yingxiang_baogao/mixed_task_radiology'
# DATASET='/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report/diag2abnormal'
# PRETRAIN='/fl-ift/med/hujunchao/models/unigpt_pro_17B'
PRETRAIN='/fl-ift/med/common/Qwen2.5-72B-Instruct'
MODEL_TYPE='qwen2'
BASE1='mixed_task_radiology_noise'
BASE2='Qwen2.5-72B-Instruct'
CKPT=/fl-ift/med/hujunchao/models/${BASE1}-${BASE2}

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset ${DATASET} \
   --input_key input \
   --output_key output \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --max_samples_train 5000000 \
   --max_samples_eval 16 \
   --pretrain ${PRETRAIN} \
   --model_type ${MODEL_TYPE} \
   --save_path ${CKPT} \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --max_epochs 5 \
   --bf16 \
   --flash_attn \
   --learning_rate 2e-5 \
   --load_ds_method custom \
   --fsdp_activation_checkpointing  \
   --max_ckpt_num 2 \
   --zero_stage 3 \
   --grad_accum_dtype fp32 \
   --gradient_checkpointing \
   --gradient_checkpointing_use_reentrant \
   --adam_offload
EOF
    # --wandb [WANDB_TOKENS]
    # --adam_offload
    # --parallel_granularity Qwen2DecoderLayer|QWenBlock、weight、decoder_layer
    # --decoder_layer_name Qwen2DecoderLayer|QWenBlock
    # --fsdp_activation_checkpointing
    # --gradient_checkpointing
    # --gradient_checkpointing_use_reentrant
    # --save_path ./checkpoint/qwen-14b-sft-fsdp 
if [[ ${1} != "slurm" ]]; then
    export PATH=$HOME/.local/bin/:$PATH
    hostfile=/etc/mpi/hostfile
    MASTER_PORT=$(shuf -n 1 -i 10000-65535)
    # torchrun --nnodes 1 --nproc_per_node 8 $training_commands
    deepspeed --hostfile $hostfile --master_port $MASTER_PORT --module $training_commands

fi