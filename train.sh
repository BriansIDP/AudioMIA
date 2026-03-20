# export CUDA_VISIBLE_DEVICES=1
pip install numpy==1.26.4
pip install qwen-omni-utils[decord] -U
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install rouge-score

torchrun --nproc_per_node=2 --master_port=12345 train.py \
  --model_name_or_path Qwen/Qwen2.5-Omni-7B \
  --dataset data/train_data.json \
  --val_dataset data/valid_data.json \
  --output_dir ./qwen25_omni_sft_out \
  --per_device_train_batch_size 3 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --num_train_epochs 5 \
  --max_seq_length 4096 \
  --save_steps 1000 \
  --logging_steps 1 \
  --gradient_checkpointing true \
  --flash_attn true \
  --bf16 true \
