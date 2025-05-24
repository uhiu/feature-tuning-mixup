# Feature Tuning Mixup (FTM)

This repository is the official PyTorch implementation of the paper: [Improving Transferable Targeted Attacks with Feature Tuning Mixup (CVPR 2025)](https://arxiv.org/abs/2411.15553).


## Preparation
This repository has been tested with Python 3.7.11, PyTorch 1.10.2, CUDA 11.3, Torchvision 0.11.3, and timm 0.5.4.

The pre-trained models are collected from torchvision and timm. They will be downloaded automatically.

Download the NIPS 2017 ImageNet-compatible dataset from [this link](https://drive.google.com/file/d/115joj6MwnKHjatXW4Yb3uoO0PTCwOmQj/view?usp=sharing).
Place the downloaded 1000 images in the `data/images/` folder. The corresponding image information is already provided in the `data/` folder.


## Usage
Run the following commands to obtain results similar to the paper.

FTM attack on ResNet50:
```bash
python main.py --model_name ResNet50 --save_dir ./exp/ResNet50/ftm --eval
```

FTM-E attack on ResNet50:
```bash
python main.py --model_name ResNet50 --save_dir ./exp/ResNet50/ftm_e --ensemble_size 2 --eval
```

Adversarial images and evaluation results will be saved in the `exp/` folder. Check the `scripts/` folder for more usage examples.


## Citation
If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{liang2025improving,
  title={Improving Transferable Targeted Attacks with Feature Tuning Mixup},
  author={Liang, Kaisheng and Dai, Xuelong and Li, Yanjie and Wang, Dong and Xiao, Bin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```


## Acknowledgement
This repository is built upon the repository of [CFM](https://github.com/dreamflake/CFM). We thank the authors for making their code publicly available.