set -x 
export MAX_JOBS=16

hostfile=/etc/mpi/hostfile
cd /fl-ift/med/jianglei/project/uniscale-dpo-support-dist-dpo-for-qwen/OpenRLHF/examples/scripts/2024_0223_qwen

# train_batch_size
# node: 8*5 = 40
node=4
gpu_per_node=8
# qwen14B，单机=4（10w条，1个epoch差不多8个小时），5机似乎4比较合适-10w条差不多1个小时
# qwen72B，5机，mbs=1，（10w条，差不多10个小时）
MICRO_BATCH=2
# 用于梯度累积，一般设为2、4、8
num_micro_batch=8
let GLOBAL_BATCH=MICRO_BATCH*node*gpu_per_node*num_micro_batch
echo $GLOBAL_BATCH

# qwen72B, meb设为1
# qwen14B, meb可以设为4
micro_eval_batch_size=1

#model_path=/fl-ift/med/common/Qwen1.5-14B-Chat/
model_path=/fl-ift/med/jianglei/project/gitcode/llama-recipes-main/src/14B_1_5_unigpt_checkpoints_pro_epoch2_78329_freez_para_lre6/hf_2/
train_dataset_path=/fl-ift/med/jianglei/data/Med_Data/DPO/exp02_2_dpo_med3w_72B_reject_label_common_2w_filter_gpt4_37314_personal_filter_task_75464.jsonl
dev_dataset_path=$train_dataset_path
chosen_key=Chosen
rejected_key=Rejected

# lr也得调整一下
#./ckpt/exp03_lr7_110B_dpo_med23k_72B_reject_110B_label_common_2w
# 这里save_steps > eval_steps，用来保障模型可以顺利保存，因为有时候eval的额外显存会导致显存溢出
read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
     --save_path ./ckpt/14B_1_5_unigpt_checkpoints_pro_epoch2_78329_freeze_para_lre6_dpo_exp02_2 \
     --save_steps 500 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size $GLOBAL_BATCH \
     --micro_train_batch_size $MICRO_BATCH \
     --micro_eval_batch_size $micro_eval_batch_size \
     --pretrain $model_path \
     --bf16 \
     --max_epochs 1 \
     --max_len 8192 \
     --zero_stage 3 \
     --beta 0.1 \
     --learning_rate 5e-7 \
     --train_dataset_path $train_dataset_path \
     --dev_dataset_path $dev_dataset_path \
     --chosen_key $chosen_key \
     --rejected_key $rejected_key \
     --flash_attn \
     --gradient_checkpointing \
     --ref_offload
EOF
     # --wandb [WANDB_TOKENS]
     # --ipo [for IPO]
     # --label_smoothing 0.1 [for cDPO]

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
if [[ ${1} != "slurm" ]]; then
    export PATH=$HOME/.local/bin/:$PATH
    nohup deepspeed --hostfile $hostfile --master_port $MASTER_PORT $training_commands >/fl-ift/med/jianglei/project/uniscale-dpo-support-dist-dpo-for-qwen/OpenRLHF/examples/scripts/2024_0223_qwen/ckpt/log.txt 2>&1 &
#    deepspeed --hostfile $hostfile --master_port $MASTER_PORT $training_commands
fi
