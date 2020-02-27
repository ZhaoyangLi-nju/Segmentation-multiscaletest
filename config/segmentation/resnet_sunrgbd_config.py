from config.default_config import DefaultConfig
import os

class SEG_RESNET_SUNRGBD_CONFIG:

    def args(self):

        ########### Quick Setup ############
        task_type = 'segmentation'
        model = 'trans2_seg_maxpool'
        arch = 'resnet'  # xception / resnet
        dataset = 'Seg_SUNRGBD'

        task_name = 'test_123'  # remove_8x4x_bottle_sgd0123 aspp_gen
        detach = False
        lr_schedule = 'lambda'  # lambda|step|plateau1|poly

        batch_size_train = 40
        batch_size_val = 40
        niter = 500
        niter_decay = 2000

        no_trans = False  # if True, no translation lossFalse
        optimizer = 'sgd'
        if optimizer == 'sgd':
            lr = 1e-2
        else:
            lr = 4e-4

        pretrained = 'imagenet'
        content_pretrained = 'imagenet'
        fix_grad = True

        content_layers = 1  # 0,1,2,3,4layer-wise semantic layers, you can change it to better adapt your task
        alpha_content = 0.5
        alpha_contrastive = 0.05
        alpha_gan = 0.1
        which_content_net = 'resnet50'  # resnet18 vgg11

        multiprocessing = True
        use_apex = True
        sync_bn = True
        gpus = [4,5,6,7]

        niter_total = niter + niter_decay
        print_freq = niter_total / 100

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]

        if no_trans:
            loss = ['CLS']
            target_modal = 'seg'
            multi_modal = False
        else:
            loss = ['CLS', 'CONTRAST']  #PIX2PIX CONTRAST SEMANTIC
            # target_modal = 'depth'
            target_modal = 'seg'
            multi_modal = False

        base_size = (256, 256)
        load_size = (420, 560)
        fine_size = (320, 420)
        filters = 'bottleneck'

        evaluate = True  # report mean acc after each epoch
        slide_windows = False

        unlabeld = False  # True for training with unlabeled data

        multi_scale = False
        # multi_targets = ['depth']
        multi_targets = ['seg']
        which_score = 'up'
        norm = 'bn'

        inference = True
        keep_fc = True
        resume = True
        # resume_path = '/home/lzy/pretrainedmodel/deeplab_v3plus_alighned_xception_coco.pth'
        # resume_path = DefaultConfig.CHECKPOINTS_DIR + '/trans2_seg_maxpool/test_COCO_baseline_resnet_imagenet_AtoB_Seg_COCO_CLS.CONTRAST_center_crop_gpus_8_Feb25_00-37-13/best_checkpoint.pth'
        resume_path = '/home/dudapeng/workspace/translate-to-seg/checkpoints/trans2_seg_maxpool/test_large_ours_resnet_imagenet_AtoB_Seg_SUNRGBD_CLS.CONTRAST_center_crop_gpus_4_Feb26_23-07-59/checkpoint.pth'

        return {

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

            'DATA_DIR_TRAIN': '/data/lzy_old/dataset/sunrgbd_seg',
            'DATA_DIR_VAL': '/data/lzy_old/dataset/sunrgbd_seg',
            # 'DATA_DIR': DefaultConfig.ROOT_DIR + '/datasets/vm_data/sunrgbd_seg',

            # MODEL
            'ARCH': arch,
            'SAVE_BEST': False,
            'NO_TRANS': no_trans,
            'DETACH': detach,
            'LOSS_TYPES': loss,

            #### DATA
            'NUM_CLASSES': 37,
            'UNLABELED': unlabeld,
            'LOAD_SIZE': load_size,
            'FINE_SIZE': fine_size,
            'BASE_SIZE': base_size,

            # TRAINING / TEST
            'RESUME': resume,
            'RESUME_PATH': resume_path,
            'LR_POLICY': lr_schedule,
            'OPTIMIZER': optimizer,
            'INFERENCE': inference,
            'FIX_GRAD': fix_grad,
            'KEEP_FC': keep_fc,

            'LR': lr,
            'NITER': niter,
            'NITER_DECAY': niter_decay,
            'NITER_TOTAL': niter_total,
            'FIVE_CROP': False,
            'EVALUATE': evaluate,
            'SLIDE_WINDOWS': slide_windows,
            'PRINT_FREQ': print_freq,

            'MULTIPROCESSING_DISTRIBUTED': multiprocessing,
            'USE_APEX': use_apex,
            'SYNC_BN': sync_bn,

            # translation task
            'WHICH_CONTENT_NET': which_content_net,
            'CONTENT_LAYERS': content_layers,
            'CONTENT_PRETRAINED': content_pretrained,
            'ALPHA_CONTENT': alpha_content,
            'ALPHA_CONTRASTIVE': alpha_contrastive,
            'ALPHA_GAN': alpha_gan,
            'TARGET_MODAL': target_modal,
            'MULTI_SCALE': multi_scale,
            'MULTI_TARGETS': multi_targets,
            'WHICH_SCORE': which_score,
            'MULTI_MODAL': multi_modal,
            'UPSAMPLE_NORM': norm,
            'MULTI_SCALE_TEST':False,#new add
            'RESIZE_LIST':[0.5,0.75,1,1.25,1.5]#new add
        }
