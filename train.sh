export CUDA_VISIBLE_DEVICES=6,7

# pip install numpy==1.26.4
# pip install qwen-omni-utils[decord] -U
# pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
# pip install rouge-score

expdir=exp/qwen25_omni_sft_newQA_lora_r512_a1024_newdata_capasr
mkdir -p $expdir

train_data=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/AudioMIA/data/newdata/newdata_train_capasr_QA.json
# train_data=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/MultimodalMIA/data/train_data.json
# train_data=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/MultimodalMIA/data/probedata/train_probe_full_with_tts.json
# train_data=data/train_data_with_tts.json
# train_data=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/MultimodalMIA/data/ttsdata.json
valid_data=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/AudioMIA/data/valid_data.json
# valid_data=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/MultimodalMIA/data/ttsdata_probe.json
# valid_data=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/MultimodalMIA/data/probedata/valid_probe.json

torchrun --nproc_per_node=1 --master_port=12345 train.py \
  --model_name_or_path Qwen/Qwen2.5-Omni-7B \
  --dataset $train_data \
  --val_dataset $valid_data \
  --output_dir $expdir \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --num_train_epochs 2 \
  --max_seq_length 4096 \
  --save_steps 1000 \
  --logging_steps 1 \
  --gradient_checkpointing true \
  --flash_attn true \
  --lora_r 512 \
  --lora_alpha 1024 \
  --bf16 true \
  # --flash_attn true \
  # --fulltune true \