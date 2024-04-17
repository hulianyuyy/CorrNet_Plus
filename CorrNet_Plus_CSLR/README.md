# CorrNet+_CSLR
This repo holds codes of the paper: CorrNet+: Sign Language Recognition and Translation via Spatial-Temporal Correlation, which is an extension of our previous work (CVPR 2023) [[paper]](https://arxiv.org/abs/2303.03202)

This sub-repo holds the code for supporting the continuous sign language recognition task with CorrNet+.

## Performance
- On the continuous sign language cognition task, CorrNet+ achieves superior performance on PHOENIX14, PHOENIX14-T, CSL-Daily and CSL datasets.

<table align="center">
<tbody align="center" valign="center">
    <tr>
        <td rowspan="3">Method</td>
        <td colspan="4">PHOENIX2014</td>
        <td colspan="2">PHOENIX2014-T</td>
        <td colspan="2">CSL-Daily</td>
    </tr>
    <tr>
        <td colspan="2">Dev(%)</td>
        <td colspan="2">Test(%)</td>
        <td rowspan="2">Dev(%)</td>
        <td rowspan="2">Test(%)</td>
        <td rowspan="2">Dev(%)</td>
        <td rowspan="2">Test(%)</td>
    </tr>
    <tr>
        <td>del/ins</td>
        <td>WER</td>
        <td>del/ins</td>
        <td>WER</td>
    </tr>
    <tr>
        <td>CVT-SLR (CVPR2023)</td>
        <td>6.4/2.6</td>
        <td>19.8</td>
        <td>6.1/2.3</td>
        <td>20.1</td>
        <td>19.4</td>
        <td>20.3</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>CoSign-2s (ICCV2023)</td>
        <td>-</td>
        <td>19.7</td>
        <td>-</td>
        <td>20.1</td>
        <td>19.5</td>
        <td>20.1</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AdaSize (PR2024)</td>
        <td>7.0/2.6</td>
        <td>19.7</td>
        <td>7.2/3.1</td>
        <td>20.9</td>
        <td>19.7</td>
        <td>21.2</td>
        <td>31.3</td>
        <td>30.9</td>
    </tr>
    <tr>
        <td>AdaBrowse+ (ACMMM2023)</td>
        <td>6.0/2.5</td>
        <td>19.6</td>
        <td>5.9/2.6</td>
        <td>20.7</td>
        <td>19.5</td>
        <td>20.6</td>
        <td>31.2</td>
        <td>30.7</td>
    </tr>
    <tr>
        <td>SEN (AAAI2023)</td>
        <td>5.8/2.6</td>
        <td>19.5</td>
        <td>7.3/4.0</td>
        <td>21.0</td>
        <td>19.3</td>
        <td>20.7</td>
        <td>31.1</td>
        <td>30.7</td>
    </tr>
    <tr>
        <td>CTCA (CVPR2023)</td>
        <td>6.2/2.9</td>
        <td>19.5</td>
        <td>6.1/2.6</td>
        <td>20.1</td>
        <td>19.3</td>
        <td>20.3</td>
        <td>31.3</td>
        <td>29.4</td>
    </tr>
    <tr>
        <td>C2SLR (CVPR2022)</td>
        <td>-</td>
        <td>20.5</td>
        <td>-</td>
        <td>20.4</td>
        <td>20.2</td>
        <td>20.4</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <th>CorrNet+</th>
        <td>5.3/2.7</td>
        <th>18.0</th>
        <td>5.6/2.4</td>
        <th>18.2</th>
        <th>17.2</th>
        <th>19.1</th>
        <th>28.6</th>
        <th>28.2</th>
    </tr>
</tbody>
</table>

- On the sign language translation task, CorrNet+ achieves superior performance on PHOENIX14, PHOENIX14-T and CSL-Daily datasets.

<table>
<tbody align="center" valign="center">
    <tr>
        <td colspan="11">PHOENIX2014-T</td>
    </tr>
    <tr>
        <td>Method</td>
        <td colspan="5">Dev(%)</td>
        <td colspan="5">Test(%)</td>
    </tr>
    <tr>
        <td></td>
        <td>Rouge</td>
        <td>BLEU1</td>
        <td>BLEU2</td>
        <td>BLEU3</td>
        <td>BLEU4</td>
        <td>Rouge</td>
        <td>BLEU1</td>
        <td>BLEU2</td>
        <td>BLEU3</td>
        <td>BLEU4</td>
    </tr>
    <tr>
        <td>SignBT (CVPR2021)</td>
        <td>50.29</td>
        <td>51.11</td>
        <td>37.90</td>
        <td>29.80</td>
        <td>24.45</td>
        <td>49.54</td>
        <td>50.80</td>
        <td>37.75</td>
        <td>29.72</td>
        <td>24.32</td>
    </tr>
    <tr>
        <td>MMTLB (CVPR2022)</td>
        <td>53.10</td>
        <td>53.95</td>
        <td>41.12</td>
        <td>33.14</td>
        <td>27.61</td>
        <td>52.65</td>
        <td>53.97</td>
        <td>41.75</td>
        <td>33.84</td>
        <td>28.39</td>
    </tr>
    <tr>
        <td>SLTUNET (ICLR2023)</td>
        <td>52.23</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>27.87</td>
        <td>52.11</td>
        <td>52.92</td>
        <td>41.76</td>
        <td>33.99</td>
        <td>28.47</td>
    </tr>
    <tr>
        <td>TwoStream-SLT (NeuIPS2023)</td>
        <td>54.08</td>
        <td>54.32</td>
        <td>41.99</td>
        <td>34.15</td>
        <td>28.66</td>
        <td>53.48</td>
        <td>54.90</td>
        <td>42.43</td>
        <td>34.46</td>
        <td>28.95</td>
    </tr>
    <tr>
        <td>CorrNet+</td>
        <th>54.54</th>
        <th>54.56</th>
        <th>42.31</th>
        <th>34.48</th>
        <th>29.13</th>
        <th>53.76</th>
        <th>55.32</th>
        <th>42.74</th>
        <th>34.86</th>
        <th>29.42</th>
    </tr>
    <tr>
        <td colspan="11">CSL-Daily</td>
    </tr>
    <tr>
        <td>Method</td>
        <td colspan="5">Dev(%)</td>
        <td colspan="5">Test(%)</td>
    </tr>
    <tr>
        <td></td>
        <td>Rouge</td>
        <td>BLEU1</td>
        <td>BLEU2</td>
        <td>BLEU3</td>
        <td>BLEU4</td>
        <td>Rouge</td>
        <td>BLEU1</td>
        <td>BLEU2</td>
        <td>BLEU3</td>
        <td>BLEU4</td>
    </tr>
    <tr>
        <td>SignBT (CVPR2021)</td>
        <td>49.49</td>
        <td>51.46</td>
        <td>37.23</td>
        <td>27.51</td>
        <td>20.80</td>
        <td>49.31</td>
        <td>51.42</td>
        <td>37.26</td>
        <td>27.76</td>
        <td>21.34</td>
    </tr>
    <tr>
        <td>MMTLB (CVPR2022)</td>
        <td>53.38</td>
        <td>53.81</td>
        <td>40.84</td>
        <td>31.29</td>
        <td>24.42</td>
        <td>53.25</td>
        <td>53.31</td>
        <td>40.41</td>
        <td>30.87</td>
        <td>23.92</td>
    </tr>
    <tr>
        <td>SLTUNET (ICLR2023)</td>
        <td>53.58</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>23.99</td>
        <td>54.08</td>
        <td>54.98</td>
        <td>41.44</td>
        <td>31.84</td>
        <td>25.01</td>
    </tr>
    <tr>
        <td>TwoStream-SLT (NeuIPS2023)</td>
        <td>55.10</td>
        <td>55.21</td>
        <td>42.31</td>
        <td>32.71</td>
        <td>25.76</td>
        <td>55.72</td>
        <td>55.44</td>
        <td>42.59</td>
        <td>32.87</td>
        <td>25.79</td>
    </tr>
    <tr>
        <td>CorrNet+</td>
        <th>55.52</th>
        <th>55.64</th>
        <th>42.78</th>
        <th>33.13</th>
        <th>26.14</th>
        <th>55.84</th>
        <th>55.82</th>
        <th>42.96</th>
        <th>33.26</th>
        <th>26.14</th>
    </tr>
</tbody>
</table>

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
   python data_preprocess.py --process-image --multiprocessing
   ```

### PHOENIX2014-T dataset
1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET/PHOENIX-2014-T-release-v3/PHOENIX-2014-T ./dataset/phoenix2014-T`

3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python data_preprocess-T.py --process-image --multiprocessing
   ```

### CSL dataset

1. Request the CSL Dataset from this website [[download link]](https://ustc-slr.github.io/openresources/cslr-dataset-2015/index.html)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET ./dataset/CSL`

3. The original image sequence is 1280x720, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python data_preprocess-CSL.py --process-image --multiprocessing
   ``` 

### CSL-Daily dataset

1. Request the CSL-Daily Dataset from this website [[download link]](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET ./dataset/CSL-Daily`

3. The original image sequence is 1280x720, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python data_preprocess-CSL-Daily.py --process-image --multiprocessing
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
`python main.py --device your_device --load-weights path_to_weight.pt --phase test`

## Training

The priorities of configuration files are: command line > config file > default values of argparse. To train the SLR model, run the command below:

`python main.py --device your_device`

Note that you can choose the target dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in line 3 in ./config/baseline.yaml.

## Visualizations
For Grad-CAM visualization, you can replace the resnet.py under "./modules" with the resnet.py under "./weight_map_generation", and then run ```python generate_cam.py``` with your own hyperparameters.
