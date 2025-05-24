import os
import argparse
from datetime import datetime

import torch
from torchvision import transforms
import numpy as np
import random
import copy
from PIL import Image
from config import exp_configuration
from utils import load_ground_truth, WrapperModel, load_model, EvalResult, save_images
from attacks import ftm_attack

now = datetime.now()
today_string = now.strftime("%Y-%m-%d|%H:%M:%S")


def main(args):
    exp_settings = exp_configuration[args.config_idx]
    device = args.device
    batch_size = args.batch_size
    img_size = args.img_size
    atk_type = args.attack_method

    # update parameters
    exp_settings['ftm_beta'] = args.beta
    exp_settings['ftm_ensemble_size'] = args.ensemble_size
    if args.debug:
        exp_settings['num_images'] = 2 * args.batch_size

    print(today_string)
    print(args)
    print(exp_settings, flush=True)
    print("#" * 50)

    seed = args.seed
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # load target models
    target_model_names = exp_settings['target_model_names']
    small_sized_models = ['vit_base_patch16_224', 'levit_384', 'convit_base', 'twins_svt_base', 'pit']
    mean, stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    trn = transforms.Compose([transforms.ToTensor(), ])
    image_id_list, label_ori_list, label_tar_list = load_ground_truth(args.image_csv)
    if args.eval:
        print("Loading target models...")
        if seed != 42:
            random.seed(seed)
            c = list(zip(image_id_list, label_ori_list, label_tar_list))
            random.shuffle(c)
            image_id_list, label_ori_list, label_tar_list = zip(*c)

        target_models = [WrapperModel(load_model(x), mean, stddev, True if x in small_sized_models else False).to(device) \
                         for x in target_model_names]
        eval_results = [EvalResult(x) for x in target_model_names]
        print("Target models loaded.", flush=True)
        print("%s target models:" % len(target_model_names), target_model_names)

    total_img_num = exp_settings['num_images']
    image_id_list = image_id_list[:total_img_num]
    label_ori_list = label_ori_list[:total_img_num]
    label_tar_list = label_tar_list[:total_img_num]

    # load source model
    source_model_name = args.model_name
    print("Loading source model", source_model_name)
    if source_model_name in small_sized_models:
        source_model = WrapperModel(load_model(source_model_name), mean, stddev, True).to(device)
    else:
        source_model = WrapperModel(load_model(source_model_name), mean, stddev).to(device)
    source_models = [source_model.eval()]
    if exp_settings['ftm_ensemble_size'] > 1:
        for _ in range(exp_settings['ftm_ensemble_size'] - 1):
            source_models.append(copy.deepcopy(source_model))
    print("Source model loaded.", flush=True)

    # perform attack
    num_batches = np.int(np.ceil(len(image_id_list) / batch_size))
    num_images = 0
    for k in range(0, num_batches):
        batch_size_cur = min(batch_size, len(image_id_list) - k * batch_size)
        img = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
        for i in range(batch_size_cur):
            img[i] = trn(Image.open(os.path.join(args.image_dir, image_id_list[k * batch_size + i] + '.png')))

        labels = torch.tensor(label_ori_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)
        target_labels = torch.tensor(label_tar_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)

        # obtain adversarial examples
        adv_imgs = ftm_attack(source_models, img, labels, target_labels, exp_settings, device, atk_type)
        num_images += batch_size_cur

        # save adversarial images
        os.makedirs(os.path.join(args.save_dir, "adv_imgs"), exist_ok=True)
        for i in range(batch_size_cur):
            img_idx = k * batch_size + i
            save_name = str(img_idx) + '_' + image_id_list[img_idx] + '_' + \
                        str(label_ori_list[img_idx]) + '_' + str(label_tar_list[img_idx]) + '.png'
            save_path = os.path.join(args.save_dir, "adv_imgs", save_name)
            save_images(adv_imgs[i], save_path)
        print(f"Saved [{num_images}/{total_img_num}] adversarial images to {args.save_dir}/adv_imgs")

        # update evaluation results
        if args.eval:
            for eval_result, target_model in zip(eval_results, target_models):
                target_model.eval()
                torch.cuda.empty_cache()
                with torch.no_grad():
                    pred_labels = target_model(adv_imgs)
                    pred_labels = torch.argmax(pred_labels, dim=1)
                    eval_result.update(pred_labels, target_labels)
                    print(eval_result)

    # save evaluation results
    if args.eval:
        eval_results_path = os.path.join(args.save_dir, "eval_results.txt")
        finish_string = datetime.now().strftime("%Y-%m-%d|%H:%M:%S")
        with open(eval_results_path, "w") as f:
            f.write(f"Evaluation Results ({finish_string})\n")
            f.write(f"Job started at {today_string}\n")
            f.write("Arguments:\n")
            for arg, value in vars(args).items():
                f.write(f"  {arg}: {value}\n")
            f.write("-" * 50 + "\n")
            for eval_result in eval_results:
                f.write(str(eval_result) + "\n")
        print(f"Evaluation results saved to {eval_results_path}")


def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device to run the attack"
    )
    parser.add_argument(
        "--attack_method",
        type=str,
        default="RTMF",
        help="the default is RDI-TI-MI-FTM"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="batch_size as an integer"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ResNet50",
        choices=['ResNet50', 'inception_v3', 'DenseNet121', 'levit_384'],
        help="source model name"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="beta parameter for FTM"
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help="1 for FTM, 2 for FTM-E"
    )
    parser.add_argument(
        "--eval",
        action='store_true',
        help="evaluation after attack"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./exp/test",
        help="path to save adversarial images and evaluation results"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="./data/images",
        help="path to input images"
    )
    parser.add_argument(
        "--image_csv",
        type=str,
        default="./data/images.csv",
        help="path to csv containing image info"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=299,
        help="image size"
    )
    parser.add_argument(
        "--config_idx",
        type=int,
        default=1,
        help="config containing default attack parameters"
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="debug mode"
    )
    return parser


if __name__ == "__main__":
    args = argument_parsing().parse_args()
    main(args)
    print("DONE")
