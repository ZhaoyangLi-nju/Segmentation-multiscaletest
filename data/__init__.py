from data.datasets import pascal, sbd, combine_dbs, coco, sbd, seg_sunrgbd, mit67, rec_sunrgbd, places, nyud2, cityscapes

def make_dataset(cfg):
    if cfg.DATASET == 'Seg_VOC':
        train_set = pascal.Seg_VOC(cfg, split='train')
        val_set = pascal.Seg_VOC(cfg, split='val')
        if cfg.USE_SBD:
            sbd_train = sbd.SBDSegmentation(cfg, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])
        test_set = None
        return train_set, val_set,test_set

    elif cfg.DATASET == 'Seg_COCO':
        train_set = coco.Seg_COCO(cfg, split='train')
        val_set = coco.Seg_COCO(cfg, split='val')
        test_set=None
        return train_set, val_set,test_set

    elif cfg.DATASET == 'Seg_Cityscapes':
        train_set = cityscapes.Seg_Cityscapes(cfg, split='train')
        val_set = cityscapes.Seg_Cityscapes(cfg, split='val')
        test_set=None
        # test_set = cityscapes.Seg_Cityscapes(cfg, split='test')
        # train_set = cityscapes.Seg_Cityscapes(cfg, split='train_extra')
        return train_set, val_set , test_set
    elif cfg.DATASET == 'Seg_SUNRGBD':
        train_set = seg_sunrgbd.Seg_SUNRGBD(cfg,split='train')
        val_set = seg_sunrgbd.Seg_SUNRGBD(cfg,split='val')
        test_set = None
        return train_set, val_set,test_set

    elif cfg.DATASET == 'Rec_SUNRGBD':
        train_set = rec_sunrgbd.Rec_SUNRGBD(cfg, split='train')
        val_set = rec_sunrgbd.Rec_SUNRGBD(cfg, split='val')
        test_set = None
        return train_set, val_set,test_set
    elif cfg.DATASET == 'Rec_MIT67':
        train_set = mit67.Rec_MIT67(cfg, split='train')
        val_set = mit67.Rec_MIT67(cfg, split='val')
        test_set = None
        return train_set, val_set, test_set
    elif cfg.DATASET == 'Rec_NYUD2':
        train_set = nyud2.Rec_NYUD2(cfg, split='train')
        val_set = nyud2.Rec_NYUD2(cfg, split='val')
        test_set = None
        return train_set, val_set, test_set
    elif cfg.DATASET == 'Rec_PLACES':
        train_set = places.Rec_PLACES(cfg, split='train')
        val_set = places.Rec_PLACES(cfg, split='val')
        test_set = None
        return train_set, val_set, test_set
    else:
        raise NotImplementedError
# from data.datasets import pascal, sbd, combine_dbs, coco, sbd, seg_sunrgbd, mit67, rec_sunrgbd, places, nyud2, \
#     cityscapes
#
#
# def make_dataset(cfg):
#     if cfg.DATASET == 'Seg_VOC':
#         train_set = pascal.Seg_VOC(cfg, split='train')
#         val_set = pascal.Seg_VOC(cfg, split='val')
#         if cfg.USE_SBD:
#             sbd_train = sbd.SBDSegmentation(cfg, split=['train', 'val'])
#             train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])
#         test_set = None
#         return train_set, val_set, test_set
#
#     elif cfg.DATASET == 'Seg_COCO':
#         train_set = coco.Seg_COCO(cfg, split='train')
#         val_set = coco.Seg_COCO(cfg, split='val')
#         test_set = None
#         return train_set, val_set, test_set
#
#     elif cfg.DATASET == 'Seg_Cityscapes':
#         train_set = cityscapes.Seg_Cityscapes(cfg, split='train')
#         val_set = cityscapes.Seg_Cityscapes(cfg, split='val')
#         test_set = cityscapes.Seg_Cityscapes(cfg, split='test')
#
#         return train_set, val_set, test_set
#     elif cfg.DATASET == 'Seg_SUNRGBD':
#         train_set = seg_sunrgbd.Seg_SUNRGBD(cfg, split='train')
#         val_set = seg_sunrgbd.Seg_SUNRGBD(cfg, split='val')
#         test_set = None
#         return train_set, val_set, test_set
#
#     elif cfg.DATASET == 'Rec_SUNRGBD':
#         train_set = rec_sunrgbd.Rec_SUNRGBD(cfg, split='train')
#         val_set = rec_sunrgbd.Rec_SUNRGBD(cfg, split='val')
#         test_set = None
#         return train_set, val_set, test_set
#     elif cfg.DATASET == 'Rec_MIT67':
#         train_set = mit67.Rec_MIT67(cfg, split='train')
#         val_set = mit67.Rec_MIT67(cfg, split='val')
#         test_set = None
#         return train_set, val_set, test_set
#     elif cfg.DATASET == 'Rec_NYUD2':
#         train_set = nyud2.Rec_NYUD2(cfg, split='train')
#         val_set = nyud2.Rec_NYUD2(cfg, split='val')
#         test_set = None
#         return train_set, val_set, test_set
#     elif cfg.DATASET == 'Rec_PLACES':
#         train_set = places.Rec_PLACES(cfg, split='train')
#         val_set = places.Rec_PLACES(cfg, split='val')
#         test_set = None
#         return train_set, val_set, test_set
#     else:
#         raise NotImplementedError
#
# # import torch.utils.data
# #
# # class DataProvider():
# #
# #     def __init__(self, cfg, dataset, batch_size=None, shuffle=True):
# #         super().__init__()
# #         self.dataset = dataset
# #         if batch_size is None:
# #             batch_size = cfg.BATCH_SIZE
# #         self.dataloader = torch.utils.data.DataLoader(
# #             self.dataset,
# #             batch_size=batch_size,
# #             shuffle=shuffle,
# #             num_workers=int(cfg.WORKERS),
# #             drop_last=False)
# #
# #     def __len__(self):
# #         return len(self.dataset)
# #
# #     def __iter__(self):
# #         for i, data in enumerate(self.dataloader):
# #             yield data
