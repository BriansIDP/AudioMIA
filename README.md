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

## Results on Full Test Set
| Method | AUROC | TPR @ Low FPR |
|---|---:|---:|
| PPL | 0.499092 | 0.050000 |
| Min-5.0% Prob | 0.510956 | 0.045556 |
| Entropy | 0.499769 | 0.050556 |
| Max Prob Gap | 0.492340 | 0.052222 |
| Max-5.0% Renyi 0.5 | 0.501353 | 0.044444 |
| Max-5.0% Renyi 1.0 | 0.506644 | 0.037778 |
| Max-5.0% Renyi 2.0 | 0.509770 | 0.041111 |
| Max-5.0% Renyi infinity | 0.510956 | 0.045556 |
| Max Sharma-Mittal | 0.511491 | 0.049444 |
| Min Sharma-Mittal | __0.514077__ | __0.061667__ |
| Mean Sharma-Mittal | 0.503708 | 0.049444 |
| sequence Sharma-Mittal | 0.504818 | 0.045556 |

## Results on Small Test Set
