# OmniFD: A Unified Model for Versatile Face Forgery Detection

This repository is the official implementation of
paper: [OmniFD: A Unified Model for Versatile Face Forgery Detection]()

## Environment

- python: 3.9
- pytorch: 2.0

```sh
conda create -n omnifd python=3.9
conda activate omnifd

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
pip install -U openmim

cd OmniFD
pip install -r requirements.txt
pip install -e .
```

## Dataset Preparation

[//]: # (The dataset directory structure is as follows:)

[//]: # (```)

[//]: # (```)


## Training and Evaluation

### Training

```shell
bash tools/dist_train.sh configs_omnifd/unified/omnifd_swin-s_50k_forgerynet.py 4
```

### Evaluation

```shell
CHECKPOINT=pretrained/omnifd_swin-s_50k_forgerynet.pth
bash tools/dist_test.sh configs_omnifd/single/omnifd_swin-s_50k_forgerynet_video.py $CHECKPOINT 4
bash tools/dist_test.sh configs_omnifd/single/omnifd_swin-s_50k_forgerynet_temporal.py $CHECKPOINT 4
bash tools/dist_test.sh configs_omnifd/single/omnifd_swin-s_50k_forgerynet_image.py $CHECKPOINT 4
bash tools/dist_test.sh configs_omnifd/single/omnifd_swin-s_50k_forgerynet_spatial.py $CHECKPOINT 4
```

### Pretrained Weights

## Acknowledgment

The code is largely based
on [MMPretrain](https://github.com/open-mmlab/mmpretrain), [MMAction2](https://github.com/open-mmlab/mmaction2), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation),
and [OpenTAD](https://github.com/sming256/OpenTAD). Thanks for their contributions.

[//]: # (## Citation)

[//]: # ()

[//]: # (If you find this repository useful in your research, please consider citing:)

[//]: # ()

[//]: # (```latex)

[//]: # (```)