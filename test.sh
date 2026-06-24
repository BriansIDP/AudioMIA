export CUDA_VISIBLE_DEVICES=0

pip install soundfile
pip install numpy==1.26.4
pip install qwen-omni-utils

expdir=exp/qwen25_omni_sft_newQA_largeLoRA_probe

# ckpt=$expdir/checkpoint-24130
# ckpt=$expdir/checkpoint-18100
ckpt=$expdir/checkpoint-2000
# ckpt=no

metrics=minkpp_completion

# evaldata=data/per_audio_all_question_testset_small_alt.json
evaldata=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/MultimodalMIA/data/probedata/valid_probe.json
# evaldata=data/train_data_qa.json
# evaldata=data/per_audio_testset_small.json
# evaldata=exp/qwen25_omni_sft_newQA/mia_qwen25omni_generation_20_samples_minkpp_all_per_audio_all_question_alt_1best.json

nsample=1

python inference.py \
    --datapath $evaldata \
    --return_logits true \
    --lora_ckpt $ckpt \
    --from_audio false \
    --output_dir $expdir \
    --n_samples $nsample \
    --metrics $metrics \
    # --change speed_noise \
    # --speed_factor 1.25 \
    # --snr_db 20 \
    # --bare_question true \

# python inference.py \
#     --datapath $evaldata \
#     --return_logits true \
#     --lora_ckpt $ckpt \
#     --from_audio false \
#     --output_dir $expdir \
#     --n_samples $nsample \
#     --metrics $metrics \
#     --change noise \
#     --snr_db 20.0 \
#     --bare_question false \