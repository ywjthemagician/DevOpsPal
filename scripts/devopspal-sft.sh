set -v 

torchrun --nproc_per_node=2 --nnodes=$WORLD_SIZE --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR --node_rank=$RANK src/train_bash.py \
    --deepspeed /mnt/llm/devopspal/Tuning/conf/deepspeed_config.json \
    --stage sft \
    --model_name_or_path /mnt/llm/devopspal/model/Qwen-7B \
    --do_train \
    --report_to 'tensorboard' \
    --dataset testing_sft \
    --template chatml \
    --finetuning_type full \
    --output_dir /mnt/llm/devopspal/model/trained \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --evaluation_strategy steps \
    --logging_steps 10 \
    --max_steps 1000 \
    --save_steps 100 \
    --eval_steps 100 \
    --learning_rate 5e-5 \
    --plot_loss \
    --max_source_length=2048 \
    --dataloader_num_workers 8 \
    --val_size 0.01 \
    --bf16 \
    --overwrite_output_dir