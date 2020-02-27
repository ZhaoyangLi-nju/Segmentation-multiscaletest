from config.recognition.resnet_sunrgbd_config import REC_RESNET_SUNRGBD_CONFIG
from config.default_config import DefaultConfig

class REC_RESNET_MIT67_CONFIG(REC_RESNET_SUNRGBD_CONFIG):

    def args(self):
        result = super().args()
        result['NUM_CLASSES'] = 67
        result['DATA_DIR_TRAIN'] = DefaultConfig.ROOT_DIR + '/datasets/indoorMIT67/train'
        result['DATA_DIR_VAL'] = DefaultConfig.ROOT_DIR + '/datasets/indoorMIT67/test'
        result['DATASET'] = 'Rec_MIT67'
        result['NITER'] = 2000
        result['NITER_DECAY'] = 8000
        result['NITER_TOTAL'] = result['NITER'] + result['NITER_DECAY']
        result['TARGET_MODAL'] = ''
        result['USE_COMPL_DATA'] = False
        result['USE_FAKE_DATA'] = False
        return result

