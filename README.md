# 1.Environment

First, create a virtual environment and activate it:

```
conda create -n new_slot python=3.8
conda activate new_slot
```

If your GPU supports CUDA 11.x, then you can use the following command to install the PyTorch:

```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```

Otherwise, you need to check the supported newest CUDA version, and install the according version of PyTorch. For example, if your GPU is NVIDIA 2080ti, then following command is recommended:

```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
```

Then, install the required packages using the following command:

```
pip install -r requirements.txt
```

>If something goes wrong, you can refer to the `environment.yml` file to get the package.

# 2. Prepare the pre-trained model

Download the pre-trained BERT model. You can use the following command (or you can directly download it from the url `https://drive.google.com/uc?id=1rCZ7ujt0q18ZDEtu-75yoT5PzCeixINI`):
```
pip install gdown
gdown https://drive.google.com/uc?id=1rCZ7ujt0q18ZDEtu-75yoT5PzCeixINI
unzip uncased_L-12_H-768_A-12.zip
```

The final directory should at least include these files:
```
./code
./data
./uncased_L-12_H-768_A-12
```

# 3. Training

Enter the `./code` directory:
```
cd code
```

First, warmup the model on 5% data:

```
CUDA_VISIBLE_DEVICES=0 bash run_script_slot/run_ours_pretrain.sh 
```

Then, train the model with different active learning strategies:
```
CUDA_VISIBLE_DEVICES=0 bash run_script_slot/run_ours_load_pretrain_XX.sh  # XX:["random","bald","entropy","margin","mmr_margin","hybrid"]
```

The models and the results will be saved in `./outputs` and `./results` respectively.

To train the model on different datasets with different parameters, change the parameters to any combination of the following, and rerun the above two commands:
>dataset = ["woz-attr", "woz-hotel", "atis"] \
alpha = [0, 0.05, 0.1, 0.15] \
beta = [0.1, 0.3, 0.5, 0.7, 0.9] 

The description of parameters:
>--strategy 'MMR_Margin' \  # The name of active learning strategy \
--labeled_ratio 0.05 \  # The ratio of warmup labeled data \
--select_ratio 0.02 \  # The ratio of unlabeled samples to be selected in each active learning iteration \
--dataset 'atis' \  # The name of the dataset \
--method 'ours' \
--setting 'semi_supervised' \
--known_cls_ratio 1.0 \
--seed 0 \  # The seed need to be fixed at 0 in all settings for the exact reproduction of results \
--backbone 'bert_MultiTask' \
--config_file_name 'ours' \
--gpu_id '0' \
--train_multitask \
--thr 0.9 \
--save_results \
--save_model \
--results_file_name 'results_ours.csv' \
--fine_tune_epoch 2 \
--alpha 0.05 \  # The proportion of weak supervision loss \
--beta 0.9  #  The proportion of uncertainty information in the bi-criteria. 𝛽 = 0 and 𝛽 = 1 are equivalent to Diversity Sampling and Margin Sampling respectively
