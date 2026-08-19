source /mnt/bn/tiktok-mm-2/miniconda3/etc/profile.d/conda.sh
conda activate stepaudio2

expdir=exp/step_audio2_sft_out_r1024_a2048_new
ckpt=$expdir/checkpoint-7975
# ckpt=no

metrics=minkpp_all
nsample=10


# evaldata=data/per_audio_all_question_testset_small_alt.json
# evaldata=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/MultimodalMIA/data/split/eval_data_gemini3_split1.json
# evaldata=exp/qwen25_omni_sft_newQA/mia_qwen25omni_generation_20_samples_minkpp_all_per_audio_all_question_alt_1best.json
nj=24

for index in {1..24}; do
    gpu_id=$(( (index - 1) / 3))
    export CUDA_VISIBLE_DEVICES=$gpu_id
    echo $gpu_id
    evaldata=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/MultimodalMIA/data/split/tmp_${nj}jobs_split$index.json
    python inference_stepaudio.py \
        --datapath $evaldata \
        --return_logits true \
        --lora_ckpt $ckpt \
        --from_audio false \
        --output_dir $expdir \
        --n_samples $nsample \
        --metrics $metrics \
        --othertag capasr_5_$index &
done
wait