from config.infomax.resnet_sunrgbd_config import INFOMAX_RESNET_SUNRGBD_CONFIG
from config.default_config import DefaultConfig

class INFOMAX_RESNET_NYUD2_CONFIG(INFOMAX_RESNET_SUNRGBD_CONFIG):

    def args(self):
        result = super().args()
        result['NUM_CLASSES'] = 10
        result['DATA_DIR_TRAIN'] = DefaultConfig.ROOT_DIR + '/datasets/nyud2/conc_data/train'
        result['DATA_DIR_VAL'] = DefaultConfig.ROOT_DIR + '/datasets/nyud2/conc_data/test'
        return result
