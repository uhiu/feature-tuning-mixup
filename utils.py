import os
import csv
import numpy as np
import PIL.Image as Image

import torchvision
import torch.nn as nn
import torch
import timm
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F


# simple wrapper model to normalize an input image
class WrapperModel(nn.Module):
    def __init__(self, model, mean, std, resize=False):
        super(WrapperModel, self).__init__()
        self.mean = torch.Tensor(mean)
        self.model = model
        self.resize = resize
        self.std = torch.Tensor(std)

    def forward(self, x):
        if self.resize == True:
            x = transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)(x)
        return self.model((x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None])

    def normalize(self, x):
        if self.resize == True:
            x = transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)(x)
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


def load_model(model_name):
    if model_name == "ResNet101":
        model = torchvision.models.resnet101(pretrained=True)
    elif model_name == 'ResNet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif model_name == 'ResNet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif model_name == 'ResNet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif model_name == "ResNet152":
        model = torchvision.models.resnet152(pretrained=True)
    elif model_name == "vgg16":
        model = torchvision.models.vgg16_bn(pretrained=True)
    elif model_name == "vgg19":
        model = torchvision.models.vgg19_bn(pretrained=True)
    elif model_name == "wide_resnet101_2":
        model = torchvision.models.wide_resnet101_2(pretrained=True)
    elif model_name == "inception_v3":
        model = torchvision.models.inception_v3(pretrained=True, transform_input=True)
    elif model_name == "resnext50_32x4d":
        model = torchvision.models.resnext50_32x4d(pretrained=True)
    elif model_name == "alexnet":
        model = torchvision.models.alexnet(pretrained=True)
    elif model_name == "mobilenet_v3_large":
        model = torchvision.models.mobilenet.mobilenet_v3_large(pretrained=True)
    elif model_name == 'DenseNet121':
        model = torchvision.models.densenet121(pretrained=True)
    elif model_name == "DenseNet161":
        model = torchvision.models.densenet161(pretrained=True)
    elif model_name == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(pretrained=True)
    elif model_name == "shufflenet_v2_x1_0":
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    elif model_name == 'GoogLeNet':
        model = torchvision.models.googlenet(pretrained=True)
    # timm models
    elif model_name == "efficientnet_b0":
        model = timm.create_model('efficientnet_b0', pretrained=True)
    elif model_name == "inception_resnet_v2":
        model = timm.create_model("inception_resnet_v2", pretrained=True)
    elif model_name == "inception_v3_timm":
        model = timm.create_model("inception_v3", pretrained=True)
    elif model_name == "inception_v4_timm":
        model = timm.create_model("inception_v4", pretrained=True)
    elif model_name == "xception":
        model = timm.create_model("xception", pretrained=True)
    # timm Transformer-based models
    elif model_name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
    elif model_name == "levit_384":
        model = timm.create_model("levit_384", pretrained=True)
    elif model_name == "convit_base":
        model = timm.create_model("convit_base", pretrained=True)
    elif model_name == "twins_svt_base":
        model = timm.create_model("twins_svt_base", pretrained=True)
    elif model_name == "pit":
        model = timm.create_model('pit_s_224', pretrained=True)
    else:
        raise ValueError(f"Not supported model name. {model_name}")
    return model


# load image metadata (Image_ID, true label, and target label)
def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append(row['ImageId'])
            label_ori_list.append(int(row['TrueLabel']) - 1)
            label_tar_list.append(int(row['TargetClass']) - 1)

    return image_id_list, label_ori_list, label_tar_list


class EvalResult:
    def __init__(self, model_name):
        self.model_name = model_name
        self.total_samples = 0
        self.success_samples = 0
        self.success_rate = 0.0

    def update(self, pred_labels, target_labels):
        batch_size = len(pred_labels)
        self.total_samples += batch_size
        self.success_samples += torch.sum(pred_labels == target_labels).item()
        self.success_rate = self.success_samples / self.total_samples * 100

    def get_results(self):
        return {
            "model_name": self.model_name,
            "total_samples": self.total_samples,
            "success_samples": self.success_samples,
            "success_rate": self.success_rate
        }

    def __str__(self):
        return f"Model: {self.model_name:<21} | Attack Success: {self.success_rate:.1f}% ({self.success_samples}/{self.total_samples})"


def save_images(img_tensor, save_path):
    img = np.array(img_tensor.cpu().detach().numpy()).transpose(1, 2, 0) * 255.
    img = img.astype(np.uint8)
    im = Image.fromarray(img)
    im.save(save_path)
