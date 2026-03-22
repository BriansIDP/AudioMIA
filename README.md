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

## Results on Small Test Set (1000 samples, 500 +ve 500 -ve)

| Method | Audio Tokens | Audio Tokens | Generated Text | Generated Text |
|---|---:|---:| ---:|---:|
|  | AUROC | TPR @ 5% FPR | AUROC | TPR @ 5% FPR |
| PPL | 0.4872 | 0.064 | 0.5091 | 0.054 |
| Min-30.0% Prob | 0.4836 | 0.056 | 0.5121 | 0.052 |
| Max Prob Gap | 0.4909 | 0.048 | 0.4959 | 0.052 |
| Max-0% Renyi 0.5 | 0.5261 | 0.068 | 0.5038 | 0.042 |
| Max-0% Renyi 1.0 (Entropy) | 0.5238 | 0.068 | 0.5124 | 0.036 |
| Max-0% Renyi 2.0 | __0.5296__ | 0.074 | 0.5144 | 0.046 |
| Max-0% Renyi infinity (Max Prob) | 0.5097 | __0.090__ | 0.5219 | 0.080 |
| Max-5% Renyi 0.5 | 0.5022 | 0.062 | 0.4984 | 0.054 |
| Max-5% Renyi 1.0 (Entropy) | 0.4883 | 0.058 | 0.5067 | 0.052 |
| Max-5% Renyi 2.0 | 0.4827 | 0.052 | 0.5102 | 0.052 |
| Max-5% Renyi infinity (Max Prob) | 0.4791 | 0.052 | 0.5081 | 0.048 |
| Max Sharma-Mittal | 0.4986 | 0.058 | --- | --- |
| Min Sharma-Mittal | 0.5185 | 0.040 | --- | --- |
| Mean Sharma-Mittal | 0.4894 | 0.050 | --- | --- |
| sequence Sharma-Mittal | 0.4937 | 0.048 | --- | --- |
