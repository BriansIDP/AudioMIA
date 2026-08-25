# pip install soundfile
pip install numpy==1.26.4
pip install qwen-omni-utils

expdir=exp/qwen25_omni_sft_newQA_lora_r512_a1024_newdata_all

# ckpt=$expdir/checkpoint-24130
# ckpt=$expdir/checkpoint-9765
# ckpt=$expdir/checkpoint-3026
ckpt=no

# metrics=minkpp_completion
metrics=minkpp_all

# evaldata=data/per_audio_all_question_testset_small_alt.json
# evaldata=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/MultimodalMIA/data/probedata/valid_probe.json
# evaldata=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/MultimodalMIA/data/ttsdata/Data/tts_probe_questions.json
# evaldata=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/AudioMIA/data/Voxpopuli/voxpopuli_train_valid.json
# evaldata=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/MultimodalMIA/data/eval_data_gemini3.json
# evaldata=exp/qwen25_omni_sft_newQA/mia_qwen25omni_generation_20_samples_minkpp_all_per_audio_all_question_alt_1best.json

nsample=10

# for index in {1..24}; do
#     gpu_id=$(( (index - 1) / 3))
#     export CUDA_VISIBLE_DEVICES=$gpu_id
#     echo $gpu_id
#     evaldata=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/AudioMIA/data/newdata/split/testdata_QA_train_24jobs_split$index.json
#     python inference.py \
#         --datapath $evaldata \
#         --return_logits true \
#         --lora_ckpt $ckpt \
#         --from_audio false \
#         --output_dir $expdir \
#         --n_samples $nsample \
#         --metrics $metrics \
#         --othertag trainQA_$index &
# done
# wait

for index in {1..12}; do
    gpu_id=$(( (index - 1) / 3 + 4))
    export CUDA_VISIBLE_DEVICES=$gpu_id
    evaldata=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/MultimodalMIA/data/split/eval_data_gemini3_testset_12jobs_split$index.json
    python inference.py \
        --datapath $evaldata \
        --return_logits true \
        --lora_ckpt $ckpt \
        --from_audio false \
        --output_dir $expdir \
        --n_samples $nsample \
        --metrics $metrics \
        --othertag unseen_$index &
done
wait
