cd /mnt/llm/devopspal/devopspalCode

set -v

python src/llmtuner/dsets/do_preprocess.py \
    --stage pt \
    --model_name_or_path path_to_model \
    --do_train \
    --dataset testing_sft \
    --template default \
    --output_dir path_to_output_checkpoint_path \
    --overwrite_cache \
    --max_source_length=2048 \
    --val_size 0.01 \
    --save_dataset_path path_to_where_you_want_to_save_preprocessed_dataset