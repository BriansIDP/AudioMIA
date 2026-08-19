pip install soundfile
pip install numpy==1.26.4
pip install qwen-omni-utils

expdir=exp/qwen25_omni_sft_newQA_r1024_a2048_probe_ref

# ckpt=$expdir/checkpoint-24130
# ckpt=$expdir/checkpoint-18695
ckpt=$expdir/checkpoint-600
# ckpt=no

metrics=minkpp_completion
# metrics=minkpp

# evaldata=data/per_audio_all_question_testset_small_alt.json
# evaldata=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/MultimodalMIA/data/probedata/valid_probe.json
# evaldata=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/MultimodalMIA/data/ttsdata/Data/tts_probe_questions.json
# evaldata=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/AudioMIA/data/Voxpopuli/voxpopuli_train_valid.json
# evaldata=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/MultimodalMIA/data/eval_data_gemini3.json
# evaldata=exp/qwen25_omni_sft_newQA/mia_qwen25omni_generation_20_samples_minkpp_all_per_audio_all_question_alt_1best.json

nsample=1
export CUDA_VISIBLE_DEVICES=0

# for index in {1..8}; do
#     gpu_id=$(( (index - 1) / 2 + 4 ))
#     export CUDA_VISIBLE_DEVICES=$gpu_id
#     echo $gpu_id
#     evaldata=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/MultimodalMIA/data/split/per_audio_testset_8jobs_split$index.json
#     python inference.py \
#         --datapath $evaldata \
#         --return_logits true \
#         --lora_ckpt $ckpt \
#         --from_audio true \
#         --output_dir $expdir \
#         --n_samples $nsample \
#         --metrics $metrics \
#         --othertag fromaudio_$index &
# done
# wait

evaldata=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/AudioMIA/data/newdata/testset_probe.json

python inference.py \
    --datapath $evaldata \
    --return_logits true \
    --lora_ckpt $ckpt \
    --from_audio false \
    --output_dir $expdir \
    --n_samples $nsample \
    --metrics $metrics \
    --othertag probe_newdata \
    # --change speed_noise \
    # --speed_factor 1.25 \
    # --snr_db 20 \
    # --bare_question true \