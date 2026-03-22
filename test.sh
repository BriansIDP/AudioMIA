export CUDA_VISIBLE_DEVICES=0

pip install soundfile
pip install numpy==1.26.4
pip install qwen-omni-utils

expdir=exp/qwen25_omni_sft_out
ckpt=$expdir/checkpoint-11675
# ckpt=no


python inference.py \
    --datapath data/per_audio_testset_small.json \
    --return_logits true \
    --lora_ckpt $ckpt \
    --from_audio true \
    --output_dir $expdir \