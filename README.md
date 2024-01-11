<div align="center">

# TEMPO-PyTorch

This is reproduction code for the paper "TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting"

</div>

<div align=center> <image src="./assets/struct.png" width="600px"> </div>

**Performance**

| dataset | multivariate | seq_len | pred_len |  lr  | layers | patch_size/stride |  mae   |
| :-----: | :---------: | :-----: | :------: | :--: | :----------------: | :---------------------: | :----: |
|  ETTH1  |    False    |   384   |    96    | 1e-3 |         6          |          16/8           |  |


**Dataset**

ETTh1, ETTh2, ETTm1, ETTm2 are put in the folder "data". PyTorch dataset classes have been implemented, you can check them in `src/dataset.py`.

**Model weights**

They would be put in the folder "gpt2" automatically or you can download files from the link: https://huggingface.co/gpt2/tree/main, and put them in folder "gpt2".

