from config.default_config import DefaultConfig
import os


class SEG_RESNET_VOC_CONFIG:

    def args(self):

        ########### Quick Setup ############
        task_type = 'segmentation'
        model = 'trans2_seg_maxpool'     # FCN UNET
        arch = 'resnet'
        dataset = 'Seg_VOC'

        task_name = 'test_voc_baseline'
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

        gpus = [0,1]  # 0,1,2,3,4,5,6,7
        batch_size_train = 12
        batch_size_val = 12

        niter = 40000
        niter_decay = 40000
        niter_total = niter + niter_decay
        print_freq = niter_total / 500

        no_trans = False  # if True, no translation loss
        if no_trans:
            loss = ['CLS']
            target_modal = None
        else:
            loss = ['CLS', 'CONTRAST'] #'CLS', 'SEMANTIC','PIX2PIX'，'CONTRAST'
            target_modal = 'seg'

        filters = 'bottleneck'
        base_size = (513, 513)
        load_size = (513, 513)
        random_scale = (1, 1.5)
        fine_size = (513, 513)
        optimizer = 'sgd'
        lr = 1e-2

        content_layers = '0,1,2,3,4'  # layer-wise semantic layers, you can change it to better adapt your task
        alpha_content = 0.1
        alpha_contrast = 0.05
        alpha_pix2pix = 10
        which_content_net = 'resnet18'

        multi_targets = ['seg']
        multi_modal = False
        which_score = 'both'
        norm = 'bn'

        evaluate = True  # report mean acc after each epoch
        slide_windows = False
        resume = False
        resume_path = 'None'
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

            'DATA_DIR_TRAIN': '/data/lzy/voc2012',
            'DATA_DIR_VAL': '/data/lzy/voc2012',
            'DATA_DIR': '/data/lzy/voc2012',

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
            'NUM_CLASSES': 21,

            # TRAINING / TEST
            'USE_SBD':True,
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