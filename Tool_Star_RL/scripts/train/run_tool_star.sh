



export PYTHONPATH=/src/verl:$PYTHONPATH
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU


bash scripts/train/train.sh \
    --train_batch_size 128 \
    --ppo_mini_batch_size 16 \
    --rollout_n 8 \
    --apply_chat True \
    --prompt_template_name re_search_template_sys \
    --actor_model_path {your_actor_model_path} \
    --project_name {your_project_name} \
    --experiment_name {your_experiment_name} \
    --nnodes 1 \
    --n_gpus_per_node 8 \
    --save_freq 10 \
    --test_freq 10 \
    --total_epochs 2 \
    --wandb_api_key {your_wandb_api_key} \
    --save_path {your_save_path} \
    --train_files {path_to_train_file}/grpo_mix_train_shuffle.parquet \
    --test_files {path_to_test_file}/grpo_mix_test.parquet
