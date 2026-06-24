export CUDA_VISIBLE_DEVICES=0
. /scratch/anaconda/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate audiomia

# pip install numpy==1.26.4
# pip install qwen-omni-utils[decord] -U
# pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
# pip install rouge-score

expdir=exp/qwen25_omni_sft_newQA_largeLoRA_rep
mkdir -p $expdir

train_data=data/train_data.json
# train_data=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/MultimodalMIA/data/probedata/train_probe_full.json
# train_data=data/train_data_with_tts.json
# train_data=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/MultimodalMIA/data/ttsdata.json
valid_data=data/valid_data.json
# valid_data=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/MultimodalMIA/data/ttsdata.json
# valid_data=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/MultimodalMIA/data/probedata/valid_probe.json

torchrun --nproc_per_node=1 --master_port=12345 train.py \
  --model_name_or_path Qwen/Qwen2.5-Omni-7B \
  --dataset $train_data \
  --val_dataset $valid_data \
  --output_dir $expdir \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --num_train_epochs 5 \
  --max_seq_length 4096 \
  --save_steps 2000 \
  --logging_steps 1 \
  --gradient_checkpointing true \
  --flash_attn true \
  --lora_r 256 \
  --lora_alpha 512 \
  --bf16 true \