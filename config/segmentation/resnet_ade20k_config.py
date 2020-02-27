from config.default_config import DefaultConfig


class RESNET_ADE20K_CONFIG:

    def args(self):
        log_dir = DefaultConfig.ROOT_DIR + '/summary/'

        ########### Quick Setup ############
        model = 'PSP'     # FCN UNET
        arch = 'resnet50'
        dataset = 'ade20k'

        task_name = 'baseline_384x512'
        lr_schedule = 'lambda'  # lambda|step|plateau1
        pretrained = 'imagenet'
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # gpus = '7'  # gpu no. you can add more gpus with comma, e.g., '0,1,2'
        gpus = '0,1,2,3'  # gpu no. you can add more gpus with comma, e.g., '0,1,2'
        batch_size_train = 8
        batch_size_val = 8

        niter = 5000
        niter_decay = 25000
        niter_total = niter + niter_decay

        no_trans = True  # if True, no translation loss
        target_modal = 'seg'
        # loss = ['CLS', 'SEMANTIC']
        loss = ['CLS']
        filters = 'bottleneck'

        load_size = (768, 1024)
        random_scale = (0.5, 2)
        fine_size = (384, 512)
        lr = 1e-2

        evaluate = True  # report mean acc after each epoch
        slide_windows = True
        content_layers = '0,1,2,3,4'  # layer-wise semantic layers, you can change it to better adapt your task
        alpha_content = 0.5
        which_content_net = 'resnet50'
        content_pretrained = 'imagenet'

        multi_targets = ['seg']
        multi_modal = False
        which_score = 'up'
        norm = 'in'

        resume = False
        # resume_path = 'PSP_None_20000.pth'
        resume_path = '/home/lzy/translate-to-seg/pspnet50/model/ade20k_50_train_epoch_100.pth'
        inference = False

        return {

            'TASK': task_name,
            'MODEL': model,
            'GPU_IDS': gpus,
            'BATCH_SIZE_TRAIN': batch_size_train,
            'BATCH_SIZE_VAL': batch_size_val,
            'PRETRAINED': pretrained,
            'DATASET': dataset,
            'MEAN': mean,
            'STD': std,

            'LOG_PATH': log_dir,
            'DATA_DIR': DefaultConfig.ROOT_DIR + '/ADEChallengeData2016/',

            'RANDOM_SCALE_SIZE': random_scale,
            'LOAD_SIZE': load_size,
            'FINE_SIZE': fine_size,
            'FILTERS': filters,

            # MODEL
            'ARCH': arch,
            'SAVE_BEST': True,
            'NO_TRANS': no_trans,
            'LOSS_TYPES': loss,

            #### DATA
            'NUM_CLASSES': 150,

            # TRAINING / TEST
            'RESUME': resume,
            'INIT_EPOCH': True,
            'RESUME_PATH': resume_path,
            'LR_POLICY': lr_schedule,
            'LR': lr,

            'NITER': niter,
            'NITER_DECAY': niter_decay,
            'NITER_TOTAL': niter_total,
            'FIVE_CROP': False,
            'EVALUATE': evaluate,
            'INFERENCE': inference,
            'SLIDE_WINDOWS': slide_windows,

            # translation task
            'WHICH_CONTENT_NET': which_content_net,
            'CONTENT_LAYERS': content_layers,
            'CONTENT_PRETRAINED': content_pretrained,
            'ALPHA_CONTENT': alpha_content,
            'TARGET_MODAL': target_modal,
            'MULTI_TARGETS': multi_targets,
            'WHICH_SCORE': which_score,
            'MULTI_MODAL': multi_modal,
            'UPSAMPLE_NORM': norm
        }

