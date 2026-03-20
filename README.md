# Official Repository for Audio Membership Inference Attack

## Data
Download data from [here](https://huggingface.co/datasets/potsawee/audio-mia-batch-20260312)
Save all audio files in `data/saved_audios`

## Shadow Model
The shadow model trained from Qwen2.5-Omni-7B can be found [here](BrianatCambridge/Qwen25OmniShadowModelAudio)
Save the downloaded model under `exp/qwen25_omni_sft_out`

## Training shadow model
`train.sh`

## Inference shadow model for MIA
`test.sh`
