export CUDA_VISIBLE_DEVICES=1

pip install soundfile
pip install numpy==1.26.4
pip install qwen-omni-utils

expdir=exp/qwen25_omni_sft_caption
ckpt=$expdir/checkpoint-9000
# ckpt=no


python inference.py \
    --datapath data/per_audio_testset_small.json \
    --return_logits true \
    --lora_ckpt $ckpt \
    --from_audio false \
    --output_dir $expdir \