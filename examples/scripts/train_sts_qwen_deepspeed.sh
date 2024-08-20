set -x
# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/fl-ift/med/hujunchao/git_root/OpenRLHF
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

DATASET='/fl-ift/med/hujunchao/git_root/OpenRLHF/data/icd10_sts/sts_score_train.xlsx#None#score,/fl-ift/med/hujunchao/git_root/OpenRLHF/data/icd10_sts/sts_class.xlsx#None#None'
PRETRAIN='/fl-ift/med/common/qwen2-7b'
BASE1='icd10_sts'
BASE2='qwen2-7b'
CKPT=ckpt/${BASE1}-${BASE2}

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sts \
   --max_len 2048 \
   --dataset ${DATASET} \
   --sentence1_key sentence1 \
   --sentence2_key sentence2 \
   --label_key score \
   --train_batch_size 128 \
   --micro_train_batch_size 8 \
   --max_samples_train 1000000 \
   --max_samples_eval 256 \
   --pretrain ${PRETRAIN} \
   --model_type qwen2_sts \
   --save_path ${CKPT} \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --max_epochs 3 \
   --bf16 \
   --flash_attn \
   --learning_rate 2e-5 \
   --load_ds_method custom \
   --max_ckpt_num 2 \
   --zero_stage 3 \
   --grad_accum_dtype fp32 \
   --gradient_checkpointing \
   --gradient_checkpointing_use_reentrant 
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
    # torchrun --nnodes 1 --nproc_per_node 8 $training_commands
    deepspeed --module $training_commands

fi
