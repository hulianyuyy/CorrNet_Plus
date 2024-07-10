# CorrNet+_CSLR
This repo holds codes of the paper: CorrNet+: Sign Language Recognition and Translation via Spatial-Temporal Correlation, which is an extension of our previous work (CVPR 2023) [[paper]](https://arxiv.org/abs/2303.03202)

This sub-repo holds the code for supporting the continuous sign language recognition task with CorrNet+.

## Prerequisites

- This project is implemented in Pytorch (better >=1.13 to be compatible with ctcdecode or these may exist errors). Thus please install Pytorch first.

- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode.

- [Optional] sclite [[kaldi-asr/kaldi]](https://github.com/kaldi-asr/kaldi), install kaldi tool to get sclite for evaluation. After installation, create a soft link toward the sclite: 
  `mkdir ./software`
  `ln -s PATH_TO_KALDI/tools/sctk-2.4.10/bin/sclite ./software/sclite`

   You may use the python version evaluation tool for convenience (by setting 'evaluate_tool' as 'python' in line 16 of ./configs/baseline.yaml), but sclite can provide more detailed statistics.

- You can install other required modules by conducting 
   `pip install -r requirements.txt`

## Implementation
The implementation for the CorrNet+ is given in [./modules/resnet.py](https://github.com/hulianyuyy/CorrNet_Plus/CorrNet_Plus_CSLR/modules/resnet.py).  

It's then equipped with after each stage in ResNet in line 195 [./modules/resnet.py](https://github.com/hulianyuyy/CorrNet_Plus/CorrNet_Plus_CSLR/modules/resnet.py).

We later found that the Identification Module with only spatial decomposition could perform on par with what we report in the paper (spatial-temporal decomposition) and is slighter faster, and thus implement it as such.

## Data Preparation
You can choose any one of following datasets to verify the effectiveness of CorrNet+.

### PHOENIX2014 dataset
1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/). Our experiments based on phoenix-2014.v3.tar.gz.

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET/phoenix2014-release ./dataset/phoenix2014`

3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python dataset_preprocess.py --process-image --multiprocessing
   ```

### PHOENIX2014-T dataset
1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET/PHOENIX-2014-T-release-v3/PHOENIX-2014-T ./dataset/phoenix2014-T`

3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python dataset_preprocess-T.py --process-image --multiprocessing
   ```

if you get an error like ```IndexError: list index out of range``` on the PHOENIX2014-T dataset, you may refer to [this issue](https://github.com/hulianyuyy/CorrNet/issues/10#issuecomment-1660363025) to tackle the problem.
### CSL dataset

1. Request the CSL Dataset from this website [[download link]](https://ustc-slr.github.io/openresources/cslr-dataset-2015/index.html)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET ./dataset/CSL`

3. The original image sequence is 1280x720, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python dataset_preprocess-CSL.py --process-image --multiprocessing
   ``` 

### CSL-Daily dataset

1. Request the CSL-Daily Dataset from this website [[download link]](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET ./dataset/CSL-Daily`

3. The original image sequence is 1280x720, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python dataset_preprocess-CSL-Daily.py --process-image --multiprocessing
   ``` 

## Inference

### PHOENIX2014 dataset

| Backbone | Dev WER  | Test WER  | Pretrained model                                             |
| -------- | ---------- | ----------- | --- |
| ResNet18 | 18.0%      | 18.2%       | [[Baidu]](https://pan.baidu.com/s/1vlCMSuqZiZkvidg4wrDlZQ?pwd=w5w9) <br />[[Google Drive]](https://drive.google.com/file/d/1jcRv4Gl98mvS4mmLH5dBU_-iN3qGq8Si/view?usp=sharing) |


### PHOENIX2014-T dataset

| Backbone | Dev WER  | Test WER  | Pretrained model                                             |
| -------- | ---------- | ----------- | --- |
| ResNet18 | 17.2%      | 19.1%       | [[Baidu]](https://pan.baidu.com/s/1PcQtWOhiTEq9RFgBZ2hWhQ?pwd=nm3c) <br />[[Google Drive]](https://drive.google.com/file/d/1uBaKoB2JaB3ydYXmpn1tv0mBZ7cAF8J9/view?usp=sharing) |

### CSL-Daily dataset

| Backbone | Dev WER  | Test WER  | Pretrained model                                            |
| -------- | ---------- | ----------- | --- |
| ResNet18 | 28.6%      | 28.2%       | [[Baidu]](https://pan.baidu.com/s/1SbulBImqn78FEYFZV5Oz1w?pwd=mx8m) <br />[[Google Drive]](https://drive.google.com/file/d/1Ve_uzEB1teTmebuQ1XAMFQ0UV0EVEGyM/view?usp=sharing) |


​	To evaluate the pretrained model, choose the dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in line 3 in ./config/baseline.yaml first, and run the command below：   
`python main.py --config ./config/baseline.yaml --device your_device --load-weights path_to_weight.pt --phase test`

## Training

The priorities of configuration files are: command line > config file > default values of argparse. To train the SLR model, run the command below:

`python main.py --config ./config/baseline.yaml --device your_device`

Note that you can choose the target dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in line 3 in ./config/baseline.yaml.

## Visualizations
For Grad-CAM visualization of spatial weight maps, you can replace the resnet.py under "./modules" with the resnet.py under "./weight_map_generation", and then run ```python generate_weight_map.py``` with your own hyperparameters. 

For Grad-CAM visualization of correlation maps, you can replace the resnet.py under "./modules" with the resnet.py under "./corr_map_generation", and then run ```python generate_corr_map.py``` with your own hyperparameters.
