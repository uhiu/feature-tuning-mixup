exp_configuration = {
    1: {
        'dataset': 'ImageNet',
        'targeted': True,
        'epsilon': 16,
        'alpha': 2,
        'max_iterations': 300,
        'num_images': 1000,
        'p': 1.,  # prob for DI

        # 'source_model_names':['ResNet50','inception_v3','DenseNet121','levit_384'],
        'target_model_names': ['ResNet18', 'ResNet50', 'vgg16', 'inception_v3', 'efficientnet_b0',
                               'DenseNet121', 'mobilenet_v2', 'inception_resnet_v2', 'inception_v4_timm', 'xception',
                               'vit_base_patch16_224', 'levit_384', 'convit_base', 'twins_svt_base', 'pit'],

        ####################################
        # FTM
        'ftm_beta': 0.01,
        'ftm_ensemble_size': 1,  # 1 for FTM, 2 for FTM-E
        'mix_prob': 0.1,
        'mix_upper_bound_feature': 0.75,

        'mixed_image_type_feature': 'C',  # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature': 'SelfShuffle',  # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature': 'M',  # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature': 0.,  # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'divisor': 4,
        'channelwise': True,
        'mixup_layer': 'conv_linear_include_last',
        #####################################
        'comment': 'Default settings for RDI-TI-MI-FTM'
    },
}
