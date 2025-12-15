
$$
/begin{flalign}
& /mathrm{/Huge SHIFT-DRP:} & /nonumber
/end{flalign}
$$

$$
/begin{flalign}
& /mathrm{/LARGE Dynamic /,Multi/,Scale/,Active/,Learning/,for/,Response} & /nonumber
/end{flalign}
$$

# Systm Requirements

```
Package                Version
---------------------- -----------
dgl-cuda11.6           0.9.1
dgllife                0.3.2
faiss-cpu              1.13.0
numpy                  1.26.4
omegaconf              2.3.0
pandas                 2.3.1
rdkit                  2023.3.2
scikit-learn           1.6.1
scipy                  1.13.1
torch                  1.13.1
torchaudio             0.13.1
torchvision            0.14.1
tqdm                   4.67.1
```



## Installation Guide

```
$ conda create -n SHIFTDRP python=3.9
$ conda activate SHIFTDRP

# You can find more versions of Torch in this link.(https://pytorch.org/get-started/previous-versions/)
$ conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
$ conda install -c dglteam dgl-cuda11.6
$ conda install -c conda-forge dgllife

$ pip install tqdm
$ pip install omegaconf
$ pip install scikit-learn
$ pip install rdkit==2023.3.2
$ pip install faiss-cpu
$ pip install numpy==1.26.4
```

# Data Preparation

```
1.Extract the file 'Data/processed_data/GDSC2_AUC.rar' to the path 'Data/processed_data/'

2.Use the data located in the 'Data/processed_data/GDSC2_AUC/split_data' path to extract drug representations using the KPGT pre-trained model (refer to https://github.com/lihan97/KPGT).
The final directory structure will be:
    Data/
    └── KPGT_npz_file/
        └── GDSC2_AUC_split_data_processed/
            ├── kpgt_GDSC2_AUC_test.npz
            ├── kpgt_GDSC2_AUC_train.npz
            └── kpgt_GDSC2_AUC_valid.npz
	
```

# RUN

```
Run SHIFT-DRP on the GDSC2 dataset: python main.py
(You can adjust the running parameters in 'config/config.yml')

The running results will be saved in: 'checkpoints/AL'
```


