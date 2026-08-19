# export CUDA_VISIBLE_DEVICES=2,3
source /mnt/bn/tiktok-mm-2/miniconda3/etc/profile.d/conda.sh
conda activate qwen3omni_gs

# pip install soundfile
# pip install --upgrade numpy
# pip install numpy==1.26.4
# pip install qwen-omni-utils


export CUDA_VISIBLE_DEVICES=0
python qwen3omni_gen_question.py 0 50 &
export CUDA_VISIBLE_DEVICES=1
python qwen3omni_gen_question.py 50 100 &
export CUDA_VISIBLE_DEVICES=2
python qwen3omni_gen_question.py 100 150 &
export CUDA_VISIBLE_DEVICES=3
python qwen3omni_gen_question.py 150 200 &
export CUDA_VISIBLE_DEVICES=4
python qwen3omni_gen_question.py 200 250 &
export CUDA_VISIBLE_DEVICES=5
python qwen3omni_gen_question.py 250 300 &

wait