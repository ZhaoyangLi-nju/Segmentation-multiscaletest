from config.default_config import DefaultConfig

class REC_RESNET_SUNRGBD_CONFIG:

    def args(self):

        ########### Quick Setup ############
        task_type = 'recognition'
        dataset = 'Rec_SUNRGBD'

        model = 'trecg_maxpool'  # trecg_maxpool
        arch = 'resnet18'  # alexnet_regular
        pretrained = 'place'
        which_direction = 'BtoA'
        multi_scale = False

        task_name = 'test_new_in'

        which_content_net = 'resnet18'
        content_pretrained = 'place'

        multiprocessing = False
        use_apex = False
        sync_bn = False
        gpus = [2,3]  # 0, 1, 2, 3, 4, 5, 6, 7
        batch_size_train = 40
        batch_size_val = 40

        no_trans = False  # if True, no translation loss
        if no_trans:
            loss = ['CLS']
            target_modal = 'depth'
        else:
            loss = ['CLS', 'CONTRAST', 'GAN']  # 'CLS', 'SEMANTIC','GAN','PIX2PIX',, 'CONTRAST'
            target_modal = 'depth'

        niter = 2000
        niter_decay = 8000
        niter_total = niter + niter_decay
        print_freq = niter_total / 100

        is_finetune = True

        # if pretrained == 'imagenet':
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        base_size = (256, 256)
        load_size = (256, 256)
        fine_size = (224, 224)
        lr = 2e-4
        filters = 'bottleneck'
        lr_schedule = 'lambda'  # lambda|step|plateau1

        unlabeld = False  # True for training with unlabeled data
        content_layers = '0,1,2,3,4'  #0,1,2,3,4 # layer-wise semantic layers, you can change it to better adapt your task
        alpha_content = 10
        alpha_contrast = 1
        alpha_gan = 0.2
        alpha_pix2pix = 10
        use_fake_data = False
        sample_model_path = 'trecg_compl/target_as_input_resnet50_AtoB_alpha_10_Rec_SUNRGBD_CLS.SEMANTIC_center_crop_gpus_4_Dec19_00-05-26.pth'
        # sample_model_path = 'trecg_compl/best_compl_resnet18_AtoB_alpha_10_Rec_SUNRGBD_CLS.SEMANTIC.GAN_center_crop_gpus_2_Dec08_14-13-59.pth'
        # if 'fake' in model:
        #     use_fake_data = True
        # else:
        #     use_fake_data = False

        norm = 'in'

        evaluate = True  # report mean acc after each epoch
        slide_windows = False
        inference = False
        resume = False
        # resume_path = 'trecg_maxpool/test_mit67_resnet50_AtoB_Rec_MIT67_CLS_center_crop_gpus_2_Dec20_20-47-43/best_checkpoint.pth'
        # resume_path = 'trecg_fake/best_sc_resnet18_BtoA_alpha_10_Rec_SUNRGBD_CLS.SEMANTIC.GAN_center_crop_gpus_2_Dec06_16-52-38.pth'
        resume_path = 'trecg/contrastive_resnet18_place_AtoB_alpha_10_Rec_SUNRGBD_CLS.SEMANTIC.CONTRAST_center_crop_gpus_1_Jan10_16-48-18/checkpoint.pth'

        dataset_train = '/data/dudapeng/datasets/sun_rgbd/data_in_class_mix/conc_data/train'
        dataset_val = '/data/dudapeng/datasets/sun_rgbd/data_in_class_mix/conc_data/test'
        # dataset_train = '/data/dudapeng/datasets/sun_rgbd/conc_jet_labeled/train'
        # dataset_val = '/data/dudapeng/datasets/sun_rgbd/conc_jet_labeled/test'

        results = {

            'TASK_TYPE': task_type,
            'TASK': task_name,
            'MODEL': model,
            'GPU_IDS': gpus,
            'BATCH_SIZE_TRAIN': batch_size_train,
            'BATCH_SIZE_VAL': batch_size_val,
            'PRETRAINED': pretrained,
            'FILTERS': filters,
            'DATASET': dataset,
            'MEAN': mean,
            'STD': std,

            # 'DATA_DIR_TRAIN': '/data/dudapeng/datasets/sun_rgbd/conc_jet',

            # 'DATA_DIR_TRAIN': '/data0/dudapeng/workspace/datasets/sun_rgbd/data_in_class_mix/conc_data/train',
            # 'DATA_DIR_VAL': '/data0/dudapeng/workspace/datasets/sun_rgbd/data_in_class_mix/conc_data/test',
            'DATA_DIR_TRAIN': dataset_train,
            'DATA_DIR_VAL': dataset_val,

            # MODEL
            'ARCH': arch,
            'SAVE_BEST': False,
            'NO_TRANS': no_trans,
            'LOSS_TYPES': loss,
            'WHICH_DIRECTION': which_direction,
            'FT': is_finetune,

            #### DATA
            'NUM_CLASSES': 19,
            'UNLABELED': unlabeld,
            'LOAD_SIZE': load_size,
            'FINE_SIZE': fine_size,
            'BASE_SIZE': base_size,
            'MULTI_TARGETS': ['depth'],
            'MULTI_SCALE': multi_scale,

            # TRAINING / TEST
            'RESUME': resume,
            'RESUME_PATH': resume_path,
            'RESUME_PATH_A': 'trecg/formal_ms_resnet18_place_AtoB_alpha_15_Rec_SUNRGBD_CLS.SEMANTIC_center_crop_gpus_1_Dec23_02-38-02/checkpoint.pth',
            'RESUME_PATH_B': 'trecg/formal_ms_resnet18_place_BtoA_alpha_15_Rec_SUNRGBD_CLS.SEMANTIC_center_crop_gpus_1_Dec23_06-50-23/checkpoint.pth',
            'SAMPLE_PATH_A': 'trecg_compl/50_resnet50_AtoB_alpha_10_Rec_SUNRGBD_CLS.SEMANTIC_center_crop_gpus_5_Dec10_13-36-14.pth',
            'SAMPLE_PATH_B': 'trecg_fake/trecg_fake_test_fake_BtoA_10000_Dec03_19-24.pth',
            'SAMPLE_MODEL_PATH': sample_model_path,
            # 'RESUME_PATH_B': 'trecg_fake/best_sc_resnet18_BtoA_alpha_10_Rec_SUNRGBD_CLS.SEMANTIC.GAN_center_crop_gpus_2_Dec06_16-52-38.pth',
            'LR_POLICY': lr_schedule,

            'LR': lr,
            'NITER': niter,
            'NITER_DECAY': niter_decay,
            'NITER_TOTAL': niter_total,
            'FIVE_CROP': False,
            'EVALUATE': evaluate,
            'SLIDE_WINDOWS': slide_windows,
            'PRINT_FREQ': print_freq,
            'INFERENCE': inference,

            'MULTIPROCESSING_DISTRIBUTED': multiprocessing,
            'USE_APEX': use_apex,
            'SYNC_BN': sync_bn,

            # translation task
            'WHICH_CONTENT_NET': which_content_net,
            'CONTENT_LAYERS': content_layers,
            'CONTENT_PRETRAINED': content_pretrained,
            'ALPHA_CONTENT': alpha_content,
            'ALPHA_CONTRASTIVE': alpha_contrast,
            'ALPHA_PIX2PIX': alpha_pix2pix,
            'ALPHA_GAN': alpha_gan,
            'TARGET_MODAL': target_modal,
            'UPSAMPLE_NORM': norm,
            'USE_FAKE_DATA': use_fake_data
        }
        return results
