set -x 

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --save_path ./checkpoint/llama3-8b-dpo \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain /fl-ift/med/common/llama3-openbiollm-8b \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 9e-6 \
   --beta 0.1 \
   --dataset OpenLLMAI/preference_dataset_mixture2_and_safe_pku \
   --ref_offload \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --gradient_checkpointing
EOF
    # --wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload 


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
