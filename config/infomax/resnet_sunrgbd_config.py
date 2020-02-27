from config.default_config import DefaultConfig

class INFOMAX_RESNET_SUNRGBD_CONFIG:

    def args(self):

        ########### Quick Setup ############
        task_type = 'infomax'
        model = 'infomax'
        arch = 'resnet18'
        dataset = 'Rec_SUNRGBD'

        task_name = 'contrastive_pixel10_homo'
        lr_schedule = 'lambda'  # lambda|step|plateau1
        pretrained = 'place'
        which_direction = 'AtoB'

        multiprocessing = False
        use_apex = False
        sync_bn = False
        gpus = [7]  # 0, 1, 2, 3, 4, 5, 6, 7
        batch_size_train = 40
        batch_size_val = 40

        niter = 2000
        niter_decay = 8000
        niter_total = niter + niter_decay
        print_freq = niter_total / 100

        no_trans = False  # if True, no translation loss
        if no_trans:
            loss = ['CLS']
        else:
            loss = ['CLS', 'CROSS', 'PIX2PIX', 'HOMO']  #'CROSS', 'PIX2PIX','HOMO'
        target_modal = 'depth'

        unlabeled = False
        is_finetune = True  # if True, finetune the backbone with downstream tasks
        evaluate = True  # report mean acc after each epoch
        resume = False
        resume_path = 'infomax/infomax_pre_diml_10000_Nov15_16-21.pth'

        multi_scale = False
        multi_scale_num = 1
        multi_targets = ['lab']

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        base_size = (256, 256)
        load_size = (256, 256)
        fine_size = (224, 224)
        lr = 2e-4
        filters = 'bottleneck'
        norm = 'bn'
        slide_windows = False

        alpha_local = 10
        alpha_prior = 0.05
        alpha_cross = 1
        alpha_homo = 10
        alpha_gan = 0.2

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

            'UNLABELED': unlabeled,
            'DATA_DIR_TRAIN': '/data/dudapeng/datasets/sun_rgbd/conc_jet_labeled/train',
            'DATA_DIR_VAL': '/data/dudapeng/datasets/sun_rgbd/conc_jet_labeled/test',
            # 'DATA_DIR_TRAIN': '/data/dudapeng/datasets/nyud2/conc_data/10k_conc',
            # 'DATA_DIR_TRAIN': '/data/dudapeng/datasets/traintest6/',
            # 'DATA_DIR_TRAIN': '/data0/dudapeng/workspace/datasets/nyud2/conc_data/10k_conc',

            # 'DATA_DIR_TRAIN': '/data0/dudapeng/workspace/datasets/sun_rgbd/data_in_class_mix/conc_data/train',
            # 'DATA_DIR_VAL': '/data0/dudapeng/workspace/datasets/sun_rgbd/data_in_class_mix/conc_data/test',
            # 'DATA_DIR_TRAIN': '/data/dudapeng/datasets/sun_rgbd/data_in_class_mix/conc_data/train',
            # 'DATA_DIR_VAL': '/data/dudapeng/datasets/sun_rgbd/data_in_class_mix/conc_data/test',

            # MODEL
            'ARCH': arch,
            'NO_TRANS': no_trans,
            'LOSS_TYPES': loss,
            'WHICH_DIRECTION': which_direction,

            #### DATA
            'NUM_CLASSES': 19,
            'LOAD_SIZE': load_size,
            'FINE_SIZE': fine_size,
            'BASE_SIZE': base_size,

            # TRAINING / TEST
            'RESUME': resume,
            'RESUME_PATH': resume_path,
            'LR_POLICY': lr_schedule,
            'FT': is_finetune,

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
            'TARGET_MODAL': target_modal,
            'MULTI_SCALE': multi_scale,
            'MULTI_SCALE_NUM': multi_scale_num,
            'MULTI_TARGETS': multi_targets,
            'ALPHA_LOCAL': alpha_local,
            'ALPHA_HOMO': alpha_homo,
            'ALPHA_PRIOR': alpha_prior,
            'ALPHA_CROSS': alpha_cross,
            'ALPHA_GAN': alpha_gan,
            'UPSAMPLE_NORM': norm
        }

