set -x
# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/fl-ift/med/hujunchao/git_root/OpenRLHF
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATASET='/fl-ift/med/hujunchao/git_root/llama-recipes-main/data/yingxiang_report/生成结论2/all_part_diagnose_fix_xuhao'
PRETRAIN='/fl-ift/med/common/ziya-llama-13b-v1'
MODEL_TYPE='llama'
BASE1='all_part_diagnose_fix_xuhao'
BASE2='ziya-llama-13b-v1'
CKPT=ckpt/${BASE1}-${BASE2}

read -r -d '' training_commands <<EOF
openrlhf/cli/train_sft.py \
   --max_len 2048 \
   --dataset ${DATASET} \
   --input_key input \
   --output_key output \
   --train_batch_size 32 \
   --micro_train_batch_size 4 \
   --max_samples_train 200000 \
   --max_samples_eval 10 \
   --pretrain ${PRETRAIN} \
   --model_type ${MODEL_TYPE} \
   --ckpt_path ${CKPT} \
   --save_steps 234 \
   --logging_steps 1 \
   --eval_steps -1 \
   --max_epochs 4 \
   --bf16 \
   --flash_attn \
   --learning_rate 2e-5 \
   --load_ds_method custom \
   --fsdp_activation_checkpointing  \
   --max_ckpt_num 2 \
   --zero_stage 3 
EOF
    # --wandb [WANDB_TOKENS]
    # --adam_offload
    # --parallel_granularity Qwen2DecoderLayer|QWenBlock、weight、decoder_layer
    # --decoder_layer_name Qwen2DecoderLayer|QWenBlock
    # --fsdp_activation_checkpointing
    # --gradient_checkpointing
    # --save_path ./checkpoint/qwen-14b-sft-fsdp 
if [[ ${1} != "slurm" ]]; then
    export PATH=$HOME/.local/bin/:$PATH
    torchrun --nnodes 1 --nproc_per_node 8 $training_commands
    # deepspeed --module $training_commands

fi