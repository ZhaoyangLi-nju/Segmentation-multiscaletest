from config.recognition.resnet_sunrgbd_config import REC_RESNET_SUNRGBD_CONFIG
from config.default_config import DefaultConfig

class REC_RESNET_NYUD2_CONFIG(REC_RESNET_SUNRGBD_CONFIG):

    def args(self):
        result = super().args()
        result['NUM_CLASSES'] = 10
        result['DATA_DIR_TRAIN'] = DefaultConfig.ROOT_DIR + '/datasets/nyud2/conc_data/train'
        result['DATA_DIR_VAL'] = DefaultConfig.ROOT_DIR + '/datasets/nyud2/conc_data/test'
        result['DATASET'] = 'Rec_NYUD2'
        result['NITER'] = 800
        result['NITER_DECAY'] = 4000
        result['NITER_TOTAL'] = result['NITER'] + result['NITER_DECAY']
        return result
