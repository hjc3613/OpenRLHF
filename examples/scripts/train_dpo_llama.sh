set -x 

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --save_path ./ckpt/qwen2_dpo \
   --save_steps -1 \
   --ckpt_path ./ckpt/qwen2_dpo \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 16 \
   --micro_train_batch_size 1 \
   --max_samples_train 10000 \
   --max_samples_eval 16 \
   --pretrain /fl-ift/med/common/Qwen2.5-72B-Instruct \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 1e-5 \
   --beta 0.1 \
   --dataset /fl-ift/med/hujunchao/datasets/comparison_data/cvalues_comparison.json \
   --ref_offload \
   --chosen_key chosen \
   --rejected_key rejected \
   --prompt_key query \
   --flash_attn \
   --gradient_checkpointing \
   --gradient_checkpointing_use_reentrant \
   --eval_steps 100 \
   --save_steps 100 \
   --max_ckpt_num 1 \
   --grad_accum_dtype fp32 \
   --load_ds_method custom \
   --model_type qwen2 \
   --freeze_strategy 'freeze:1-40-1' 
EOF
    # --wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload 


if [[ ${1} != "slurm" ]]; then
    hostfile=/etc/mpi/hostfile
    deepspeed --hostfile $hostfile --module $training_commands
fi
