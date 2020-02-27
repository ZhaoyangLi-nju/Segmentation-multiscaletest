from config.default_config import DefaultConfig
import os


class SEG_RESNET_CITYSCAPE_CONFIG:

    def args(self):

        ########### Quick Setup ############
        task_type = 'segmentation'
        model = 'trans2_seg_maxpool'     # FCN UNET
        arch = 'resnet'
        dataset = 'Seg_Cityscapes'

        task_name = 'formal_ours_01_0.5'
        lr_schedule = 'poly'  # lambda|step|plateau1
        pretrained = 'imagenet'
        content_pretrained = 'imagenet'
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # mean = [item * 255 for item in mean]
        # std = [item * 255 for item in std]

        multiprocessing = True
        use_apex = True
        sync_bn = True

        gpus = [3,4,5]  # 0,1,2,3,4,5,6,7
        batch_size_train = 16
        batch_size_val = 8

        niter = 1000
        niter_decay = 20000
        niter_total = niter + niter_decay
        print_freq = niter_total / 500

        no_trans = False # if True, no translation loss
        if no_trans:
            loss = ['CLS']
            target_modal = None
        else:
            loss = ['CLS', 'CONTRAST'] #'CLS', 'SEMANTIC','PIX2PIX'，'CONTRAST'
            target_modal = 'seg'

        filters = 'bottleneck'
        base_size = (1024, 2048)
        load_size = (1024, 2048)
        random_scale = (0.5, 2)
        fine_size = (784, 784)
        optimizer = 'sgd'
        lr = 1e-2

        content_layers = 1  # layer-wise semantic layers, you can change it to better adapt your task
        alpha_content = 0.1
        alpha_contrast = 0.5
        alpha_pix2pix = 10
        which_content_net = 'resnet18'

        multi_targets = ['seg']
        multi_modal = False
        which_score = 'both'
        norm = 'bn'

        evaluate = True  # report mean acc after each epoch
        slide_windows = False
        resume = False
        # resume_path = 'PSP_None_20000.pth'
        resume_path = 'psp_resnet50_train_epoch_200.pth'
        inference = False
        save_best = True

        return {

            'TASK_TYPE': task_type,
            'TASK': task_name,
            'MODEL': model,
            'GPU_IDS': gpus,
            'BATCH_SIZE_TRAIN': batch_size_train,
            'BATCH_SIZE_VAL': batch_size_val,

            'PRETRAINED': pretrained,
            'DATASET': dataset,
            'MEAN': mean,
            'STD': std,

            'DATA_DIR': '/data/lzy_old/dataset/cityscapes',
            # 'DATA_DIR': '/data1/lzy/',
            # 'DATA_DIR_VAL': '/data/lzy_old/dataset/cityscapes',
            # 'DATA_DIR': DefaultConfig.ROOT_DIR + '/datasets/vm_data/cityscapes',

            'BASE_SIZE': base_size,
            'RANDOM_SCALE_SIZE': random_scale,
            'LOAD_SIZE': load_size,
            'FINE_SIZE': fine_size,
            'FILTERS': filters,

            # MODEL
            'ARCH': arch,
            'SAVE_BEST': save_best,
            'NO_TRANS': no_trans,
            'LOSS_TYPES': loss,

            #### DATA
            'NUM_CLASSES': 19,

            # TRAINING / TEST
            'RESUME': resume,
            'INIT_EPOCH': True,
            'RESUME_PATH': resume_path,
            'LR_POLICY': lr_schedule,
            'OPTIMIZER': optimizer,
            'LR': lr,
            'MULTIPROCESSING_DISTRIBUTED': multiprocessing,
            'USE_APEX': use_apex,
            'SYNC_BN': sync_bn,

            'NITER': niter,
            'NITER_DECAY': niter_decay,
            'NITER_TOTAL': niter_total,
            'FIVE_CROP': False,
            'EVALUATE': evaluate,
            'INFERENCE': inference,
            'SLIDE_WINDOWS': slide_windows,
            'PRINT_FREQ': print_freq,


            # translation task
            'WHICH_CONTENT_NET': which_content_net,
            'CONTENT_LAYERS': content_layers,
            'CONTENT_PRETRAINED': content_pretrained,
            'ALPHA_CONTENT': alpha_content,
            'ALPHA_PIX2PIX': alpha_pix2pix,
            'ALPHA_CONTRASTIVE': alpha_contrast,
            'TARGET_MODAL': target_modal,
            'MULTI_TARGETS': multi_targets,
            'WHICH_SCORE': which_score,
            'MULTI_MODAL': multi_modal,
            'UPSAMPLE_NORM': norm,
            'MULTI_SCALE_TEST':True,
            'RESIZE_LIST':[0.5,0.75,1,1.25,1.5]
        }

