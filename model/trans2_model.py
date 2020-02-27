import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torchvision

import util.utils as util
from util.average_meter import AverageMeter
from . import networks as networks
from .base_model import BaseModel
from tqdm import tqdm
import cv2
import copy
import apex
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
import random
from util.metrics import Evaluator
# from thop import profile


class Trans2Net(BaseModel):

    def __init__(self, cfg, writer=None, batch_norm=nn.BatchNorm2d):
        super().__init__(cfg)
        self.phase = cfg.PHASE
        self.trans = not cfg.NO_TRANS
        self.content_model = None
        self.writer = writer
        self.batch_size_train = cfg.BATCH_SIZE_TRAIN
        self.batch_size_val = cfg.BATCH_SIZE_VAL
        self.batch_norm = batch_norm
        self._define_networks()
        self.params_list = []
        self.evaluator = Evaluator(cfg.NUM_CLASSES)
        # self.set_criterion(cfg)

    def _define_networks(self):

        networks.batch_norm = self.batch_norm
        self.net = networks.define_netowrks(self.cfg, device=self.device)

        self.model_names = ['net']

        if 'GAN' in self.cfg.LOSS_TYPES:
            self.discriminator = networks.GANDiscriminator(self.cfg, device=self.device)
            # self.discriminator = networks.GANDiscriminator_Image(self.cfg, device=self.device)
            self.model_names.append('discriminator')

        # if 'PSP' in cfg.MODEL:
        #     self.modules_ft = [self.net.layer0, self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4]
        #     self.modules_sc = [self.net.ppm, self.net.cls, self.net.aux, self.net.score_aux1, self.net.score_aux2]
        #
        #     if self.trans:
        #         self.modules_ft.extend(
        #             [self.net.up0, self.net.up1, self.net.up2, self.net.up3,
        #              self.net.up4, self.net.up5, self.net.up_seg])
        #
        #     for module in self.modules_sc:
        #         self.params_list.append(dict(params=module.parameters(), lr=cfg.LR * 5))
        #     for module in self.modules_ft:
        #         self.params_list.append(dict(params=module.parameters(), lr=cfg.LR))

        # else:
        #
        #     self.modules_ft = [self.net.layer0, self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4]
        #     self.modules_sc = [self.net.score_head, self.net.score_aux1, self.net.score_aux2]
        #     if self.trans:
        #         self.modules_sc.extend(
        #             [self.net.up1, self.net.up2, self.net.up3, self.net.up4, self.net.up5, self.net.up_image])

        # for module in self.modules_sc:
        #     self.params_list.append(dict(params=module.parameters(), lr=cfg.LR * 5))
        # for module in self.modules_ft:
        #     self.params_list.append(dict(params=module.parameters(), lr=cfg.LR))

        if self.cfg.USE_FAKE_DATA or self.cfg.USE_COMPL_DATA:
            print('Use fake data: sample model is {0}'.format(self.cfg.SAMPLE_MODEL_PATH))
            print('fake ratio:', self.cfg.FAKE_DATA_RATE)
            cfg_sample = copy.deepcopy(self.cfg)
            cfg_sample.USE_FAKE_DATA = False
            cfg_sample.USE_COMPL_DATA = False
            cfg_sample.NO_TRANS = False
            # cfg_sample.ARCH = 'resnet18'
            cfg_sample.MODEL = 'trecg_compl'
            model = networks.define_netowrks(cfg_sample, device=self.device)
            checkpoint_path = os.path.join(self.cfg.CHECKPOINTS_DIR, self.cfg.SAMPLE_MODEL_PATH)
            self._load_checkpoint(model, checkpoint_path, key='net', keep_fc=False)

            # for mit 67
            # self.net = copy.deepcopy(model.compl_net)

            model.eval()
            if self.cfg.USE_COMPL_DATA:
                self.net.set_sample_model(model)
            else:
                self.sample_model = nn.DataParallel(model).to(self.device)

        # networks.print_network(self.net)

        # print('Use fake data: sample model is {0}'.format(cfg.SAMPLE_MODEL_PATH))
        # print('fake ratio:', cfg.FAKE_DATA_RATE)
        # sample_model_path = cfg.SAMPLE_MODEL_PATH
        # cfg_sample = copy.deepcopy(cfg)
        # cfg_sample.USE_FAKE_DATA = False
        # model = networks.define_netowrks(cfg_sample, device=self.device)
        # self.load_checkpoint(net=model, checkpoint_path=sample_model_path)
        # model.eval()
        # self.sample_model = nn.DataParallel(model).to(self.device)

    def set_device(self):

        if not self.cfg.MULTIPROCESSING_DISTRIBUTED:
            self.net = nn.DataParallel(self.net).to(self.device)
            if 'GAN' in self.cfg.LOSS_TYPES:
                self.discriminator = nn.DataParallel(self.discriminator).to(self.device)

    def _optimize(self, iter):

        self._forward()

        if 'GAN' in self.cfg.LOSS_TYPES:

            self.set_requires_grad(self.net, False)
            self.set_requires_grad(self.discriminator, True)
            # fake_d = self.result['gen_img']
            # real_d = self.target_modal
            fake_d = self.result['feat_gen']['4']
            real_d = self.result['feat_target']['4']
            # fake_d = torch.cat([self.result['feat_gen']['2'], self.result['feat_target']['2']], 1)
            # real_d = torch.cat([self.result['feat_target']['2'], self.result['feat_target']['2']], 1)

            if self.cfg.MULTIPROCESSING_DISTRIBUTED:
                loss_d_fake = self.discriminator(fake_d.detach(), False)
                loss_d_true = self.discriminator(real_d.detach(), True)
            else:
                loss_d_fake = self.discriminator(fake_d.detach(), False).mean()
                loss_d_true = self.discriminator(real_d.detach(), True).mean()

            loss_d = (loss_d_fake + loss_d_true) * 0.5
            self.loss_meters['TRAIN_GAN_D_LOSS'].update(loss_d.item(), self.batch_size)

            self.optimizer_d.zero_grad()
            if self.cfg.USE_APEX and self.cfg.MULTIPROCESSING_DISTRIBUTED:
                with apex.amp.scale_loss(loss_d, self.optimizer_d) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_d.backward()
            self.optimizer_d.step()

        loss_g = self._construct_loss(iter)

        if 'GAN' in self.cfg.LOSS_TYPES and self.discriminator is not None:
            self.set_requires_grad(self.discriminator, False)
            self.set_requires_grad(self.net, True)

        self.optimizer.zero_grad()
        if self.cfg.USE_APEX and self.cfg.MULTIPROCESSING_DISTRIBUTED:
            with apex.amp.scale_loss(loss_g, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_g.backward()
        self.optimizer.step()

    def set_criterion(self, cfg):

        if 'CLS' in self.cfg.LOSS_TYPES or self.cfg.EVALUATE:
            criterion_cls = util.CrossEntropyLoss(weight=cfg.CLASS_WEIGHTS_TRAIN, device=self.device,
                                                  ignore_index=cfg.IGNORE_LABEL)
            self.net.set_cls_criterion(criterion_cls)

        if 'SEMANTIC' in self.cfg.LOSS_TYPES:
            criterion_content = torch.nn.L1Loss()
            content_model = networks.Content_Model(cfg, criterion_content).to(self.device)
            self.net.set_content_model(content_model)

        if 'PIX2PIX' in self.cfg.LOSS_TYPES:
            criterion_pix2pix = torch.nn.L1Loss()
            self.net.set_pix2pix_criterion(criterion_pix2pix)


    def set_input(self, data):

        self._source = data['image']
        self.source_modal = self._source.to(self.device)
        self.batch_size = self._source.size()[0]
        if 'label' in data.keys():
            self._label = data['label']
            self.label = torch.LongTensor(self._label).to(self.device)
        else:
            self.label = None

        if self.cfg.TARGET_MODAL:
            if self.cfg.MULTI_SCALE:
                self.target_modal = data[self.cfg.TARGET_MODAL][-1].to(self.device)
            else:
                self.target_modal = data[self.cfg.TARGET_MODAL].to(self.device)
        else:
            self.target_modal = None

        # if self.trans or self.cfg.RESUME:
        #     if not self.cfg.MULTI_SCALE:
        #         self.target_modal = self.target_modal
        #     else:

        # if self.cfg.WHICH_DIRECTION == 'BtoA':
        #     self.source_modal, self.target_modal = self.target_modal, self.source_modal

    def train_parameters(self, cfg):

        assert self.cfg.LOSS_TYPES
        self.set_optimizer(cfg)
        self.set_log_data(cfg)
        self.set_schedulers(cfg)
        self.set_device()

        # self.net = nn.DataParallel(self.net).to(self.device)

        train_iters = 0
        best_result = 0

        if self.cfg.EVALUATE and self.cfg.SLIDE_WINDOWS:
            self.prediction_matrix = torch.zeros(self.batch_size_val, self.cfg.NUM_CLASSES, self.cfg.BASE_SIZE[0],
                                                 self.cfg.BASE_SIZE[1]).to(self.device)
            self.count_crop_matrix = torch.zeros(self.batch_size_val, 1, self.cfg.BASE_SIZE[0],
                                                 self.cfg.BASE_SIZE[1]).to(
                self.device)

        if cfg.INFERENCE:
            for key in self.loss_meters:
                self.loss_meters[key].reset()
            self.phase = 'test'
            start_time = time.time()
            print('Inferencing model...')
            self.evaluate()
            self.print_evaluate_results()
            print('Evaluation Time: {0} sec'.format(time.time() - start_time))
            self.write_loss(phase=self.phase)
            return

        if cfg.MULTIPROCESSING_DISTRIBUTED:
            total_epoch = int(
                cfg.NITER_TOTAL / math.ceil((self.train_image_num / (cfg.BATCH_SIZE_TRAIN * len(cfg.GPU_IDS)))))
        else:
            total_epoch = int(cfg.NITER_TOTAL / math.ceil((self.train_image_num / cfg.BATCH_SIZE_TRAIN)))

        print('total epoch:{0}, total iters:{1}'.format(total_epoch, cfg.NITER_TOTAL))

        for epoch in range(cfg.START_EPOCH, total_epoch + 1):

            if train_iters > cfg.NITER_TOTAL:
                break

            if cfg.MULTIPROCESSING_DISTRIBUTED:
                cfg.train_sampler.set_epoch(epoch)

            self.print_lr()

            # current_lr = util.poly_learning_rate(cfg.LR, train_iters, cfg.NITER_TOTAL, power=0.8)

            # if cfg.LR_POLICY != 'plateau':
            #     self.update_learning_rate(step=train_iters)
            # else:
            #     self.update_learning_rate(val=self.loss_meters['VAL_CLS_LOSS'].avg)

            self.fake_image_num = 0

            torch.cuda.empty_cache()
            start_time = time.time()

            self.phase = 'train'
            self.net.train()

            # reset Averagemeters on each epoch
            for key in self.loss_meters:
                self.loss_meters[key].reset()

            iters = 0
            print('gpu_ids:', cfg.GPU_IDS)
            print('# Training images num = {0}'.format(self.train_image_num))
            # batch = tqdm(self.train_loader)
            # for data in batch:
            # for i, data in enumerate(batch):
            for data in self.train_loader:
                self.set_input(data)
                train_iters += 1
                iters += 1

                self._optimize(train_iters)
                self.update_learning_rate(step=train_iters)

                # self.val_iou = self.validate(train_iters)
                # self.write_loss(phase=self.phase, global_step=train_iters)

            print('log_path:', cfg.LOG_PATH)
            print('iters in one epoch:', iters)
            self.write_loss(phase=self.phase, global_step=train_iters)
            print('Epoch: {epoch}/{total}'.format(epoch=epoch, total=total_epoch))
            util.print_current_errors(util.get_current_errors(self.loss_meters, current=False), epoch)
            torch.cuda.synchronize()
            print('Training Time: {0} sec'.format(time.time() - start_time))

            # if cfg.EVALUATE:
            if (epoch % self.cfg.EVALUATE_FREQ == 0 or epoch > total_epoch - 5 or epoch == total_epoch) and cfg.EVALUATE:
                print('# Cls val images num = {0}'.format(self.val_image_num))
                self.evaluate()
                self.print_evaluate_results()
                self.write_loss(phase=self.phase, global_step=train_iters)

                # save best model
                if cfg.SAVE_BEST and epoch > total_epoch - 5:
                    # save model
                    model_filename = 'best_checkpoint.pth'
                    if cfg.TASK_TYPE == 'segmentation':
                        key = 'MEAN_IOU'
                    else:
                        key = 'MEAN_ACC'
                    for k in self.loss_meters:
                        if key in k and self.loss_meters[k].val > 0:
                            if self.loss_meters[k].val > best_result:
                                best_result = self.loss_meters[k].val
                                print('best epoch / iters are {0}/{1}'.format(epoch, iters))
                                self.save_checkpoint(model_filename)
                                if self.cfg.TASK_TYPE == 'recognition':
                                    np.savetxt(os.path.join(self.save_dir, 'class_acc.txt'), self.accuracy_class)
                                    util.plot_confusion_matrix(self.target_index_all, self.pred_index_all,
                                                               self.val_loader.dataset.classes, os.path.join(self.save_dir, 'confusion_matrix.png'))
                                break
                    print('best {0} is {1}, epoch is {2}, iters {3}'.format(key, best_result, epoch, iters))


            print('End of iter {0} / {1} \t '
                  'Time Taken: {2} sec'.format(train_iters, cfg.NITER_TOTAL, time.time() - start_time))
            print('-' * 80)

            torch.cuda.empty_cache()

    def evaluate(self):

        if not self.cfg.SLIDE_WINDOWS:
            self.validate()
        else:
            self.validate_slide_window()

    def save_best(self, best_result, epoch=None, iters=None):

        if 'segmentation' == self.cfg.TASK_TYPE:
            result = self.loss_meters['VAL_CLS_MEAN_IOU'].val
        elif 'recognition' == self.cfg.TASK_TYPE or 'infomax' == self.cfg.TASK_TYPE:
            result = self.loss_meters['VAL_CLS_MEAN_ACC'].val

        is_best = result > best_result
        best_result = max(result, best_result)
        if is_best:
            model_filename = 'best_{0}.pth'.format(self.cfg.LOG_NAME)
            print('best epoch / iters are {0}/{1}'.format(epoch, iters))
            self.save_checkpoint(model_filename)
            util.plot_confusion_matrix(self.target_index_all, self.pred_index_all, self.save_dir,
                                  self.val_loader.dataset.classes)
            print('best miou is {0}, epoch is {1}, iters {2}'.format(best_result, epoch, iters))

    def print_evaluate_results(self):
        if self.cfg.TASK_TYPE == 'segmentation':
            print(
                'MIOU: {miou}, mAcc: {macc}, acc: {acc}'.format(miou=self.loss_meters[
                                                                         'VAL_CLS_MEAN_IOU'].val * 100,
                                                                macc=self.loss_meters[
                                                                         'VAL_CLS_MEAN_ACC'].val * 100,
                                                                acc=self.loss_meters[
                                                                        'VAL_CLS_ACC'].val * 100))

        elif self.cfg.TASK_TYPE == 'recognition' or self.cfg.TASK_TYPE == 'infomax':
            print('Mean Acc Top1 <{mean_acc:.3f}> '.format(mean_acc=self.loss_meters[
                                                                        'VAL_CLS_MEAN_ACC'].val * 100))

    def _forward(self, cal_loss=True):

        if self.cfg.USE_FAKE_DATA:
            with torch.no_grad():
                result_sample = self.sample_model(source=self.source_modal, target=None, label=None, phase=self.phase,
                               cal_loss=False)

            fake_imgs = result_sample['gen_img']
            input_num = len(fake_imgs)
            indexes = [i for i in range(input_num)]
            random_index = random.sample(indexes, int(len(fake_imgs) * self.cfg.FAKE_DATA_RATE))

            for i in random_index:
                self.source_modal[i, :] = fake_imgs.data[i, :]
        if self.cfg.MULTI_SCALE_TEST and self.phase=='test':#new add
            scores = torch.zeros(1,self.cfg.NUM_CLASSES, self.cfg.LOAD_SIZE[0], self.cfg.LOAD_SIZE[1]).to(self.device)
            for i in range(len(self.cfg.RESIZE_LIST)):
                self.source_modal_multi = self.data['image_multiscale'][i].to(self.device)
                self.result = self.net(source=self.source_modal_multi, target=self.target_modal, label=self.label, phase=self.phase,
                                   cal_loss=cal_loss,segSize=self.cfg.LOAD_SIZE)
                scores = scores + self.result['cls']/len(self.cfg.RESIZE_LIST)

            # _, pred = torch.max(scores, dim=1)
            # self.result['cls'] = pred.squeeze(0).data.cpu().numpy()
            self.result['cls'] = scores

        else:
            self.result = self.net(source=self.source_modal, target=self.target_modal, label=self.label, phase=self.phase,
                               cal_loss=cal_loss)
        if self.trans:
            self.gen = self.result['gen_img']

    def _construct_loss(self, iter):

        loss_total = torch.zeros(1).to(self.device)

        if 'CLS' in self.cfg.LOSS_TYPES:

            if self.cfg.MULTIPROCESSING_DISTRIBUTED:
                cls_loss = self.result['loss_cls'] * self.cfg.ALPHA_CLS
                loss_total += cls_loss

                dist.all_reduce(cls_loss)

                if 'compl' in self.cfg.MODEL or self.cfg.USE_COMPL_DATA:
                    cls_loss_compl = self.result['loss_cls_compl'] * self.cfg.ALPHA_CLS
                    loss_total += cls_loss_compl

                    # cls_loss_fuse = self.result['loss_cls_fuse'] * self.cfg.ALPHA_CLS
                    # loss_total += cls_loss_fuse

                    dist.all_reduce(cls_loss_compl)

            else:
                cls_loss = self.result['loss_cls'].mean() * self.cfg.ALPHA_CLS
                loss_total += cls_loss

                if 'compl' in self.cfg.MODEL or self.cfg.USE_COMPL_DATA:
                    cls_loss_compl = self.result['loss_cls_compl'].mean() * self.cfg.ALPHA_CLS
                    loss_total += cls_loss_compl
                    # cls_loss_fuse = self.result['loss_cls_fuse'].mean() * self.cfg.ALPHA_CLS
                    # loss_total += cls_loss_fuse

            self.loss_meters['TRAIN_CLS_LOSS'].update(cls_loss.item(), self.batch_size)
            if 'compl' in self.cfg.MODEL or self.cfg.USE_COMPL_DATA:
                self.loss_meters['TRAIN_CLS_LOSS_COMPL'].update(cls_loss_compl.item(), self.batch_size)
                # self.loss_meters['TRAIN_CLS_LOSS_FUSE'].update(cls_loss_fuse.item(), self.batch_size)

        # ) content supervised
        if 'SEMANTIC' in self.cfg.LOSS_TYPES:

            # if self.cfg.MULTI_MODAL:
            #     self.gen = [self.result['gen_img_1'], self.result['gen_img_2']]
            # else:
            decay_coef = 1
            # decay_coef = (iters / self.cfg.NITER_TOTAL)  # small to big
            # decay_coef = max(0, (self.cfg.NITER_TOTAL - iter) / self.cfg.NITER_TOTAL) # big to small
            if self.cfg.MULTIPROCESSING_DISTRIBUTED:
                content_loss = self.result['loss_content'] * self.cfg.ALPHA_CONTENT * decay_coef
                loss_total += content_loss

                dist.all_reduce(content_loss)
                # content_loss = content_loss.detach() / self.batch_size

            else:
                content_loss = self.result['loss_content'].mean() * self.cfg.ALPHA_CONTENT * decay_coef
                loss_total += content_loss

            self.loss_meters['TRAIN_SEMANTIC_LOSS'].update(content_loss.item(), self.batch_size)

        if 'CONTRAST' in self.cfg.LOSS_TYPES:
            decay_coef = 1
            # decay_coef = (iters / self.cfg.NITER_TOTAL)  # small to big
            # decay_coef = max(0.1, (self.cfg.NITER_TOTAL - iter) / self.cfg.NITER_TOTAL) # big to small
            if self.cfg.MULTIPROCESSING_DISTRIBUTED:
                contrast_loss = self.result['loss_contrast'] * self.cfg.ALPHA_CONTRASTIVE * decay_coef
                loss_total += contrast_loss
                dist.all_reduce(contrast_loss)

            else:
                contrast_loss = self.result['loss_contrast'].mean() * self.cfg.ALPHA_CONTRASTIVE * decay_coef
                loss_total += contrast_loss

            self.loss_meters['TRAIN_CONTRAST_LOSS'].update(contrast_loss.item(), self.batch_size)

        if 'PIX2PIX' in self.cfg.LOSS_TYPES:

            decay_coef = 1
            if self.cfg.MULTIPROCESSING_DISTRIBUTED:
                pix2pix_loss = self.result['loss_pix2pix'] * self.cfg.ALPHA_PIX2PIX * decay_coef
                loss_total += pix2pix_loss
            else:
                pix2pix_loss = self.result['loss_pix2pix'].mean() * self.cfg.ALPHA_PIX2PIX * decay_coef
                loss_total += pix2pix_loss

            self.loss_meters['TRAIN_PIX2PIX_LOSS'].update(pix2pix_loss, self.batch_size)

        if 'GAN' in self.cfg.LOSS_TYPES:

            # real_g = self.result['gen_img']
            real_g = self.result['feat_gen']['4']
            # real_g = torch.cat((self.result['feat_gen']['4'], self.result['feat_gen']['4']), 1)
            if self.cfg.MULTIPROCESSING_DISTRIBUTED:
                loss_gan_g = self.discriminator(real_g, True) * self.cfg.ALPHA_GAN
            else:
                loss_gan_g = self.discriminator(real_g, True).mean() * self.cfg.ALPHA_GAN
            self.loss_meters['TRAIN_GAN_G_LOSS'].update(loss_gan_g.item(), self.batch_size)

            loss_total += loss_gan_g

        return loss_total

    def set_log_data(self, cfg):

        self.loss_meters = defaultdict()
        self.log_keys = [
            'TRAIN_GAN_G_LOSS',
            'TRAIN_GAN_D_LOSS',
            'TRAIN_SEMANTIC_LOSS',  # semantic
            'TRAIN_PIX2PIX_LOSS',
            'TRAIN_CONTRAST_LOSS',
            'TRAIN_CLS_ACC',
            'TRAIN_CLS_LOSS',
            'TRAIN_CLS_MEAN_IOU',
            'VAL_CLS_ACC',  # classification
            'VAL_CLS_LOSS',
            'VAL_CLS_MEAN_IOU',
            'VAL_CLS_MEAN_ACC',
            'INTERSECTION',
            'UNION',
            'LABEL',
            'TRAIN_CLS_LOSS_COMPL',
            'TRAIN_CLS_LOSS_FUSE'
        ]
        for item in self.log_keys:
            self.loss_meters[item] = AverageMeter()

    def set_optimizer(self, cfg):

        self.optimizers = []

        if self.cfg.OPTIMIZER == 'sgd':
            train_params = [{'params': self.net.get_1x_lr_params(), 'lr': cfg.LR},
                            {'params': self.net.get_10x_lr_params(), 'lr': cfg.LR * 10}]
            self.optimizer = torch.optim.SGD(train_params, lr=cfg.LR, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
        else:
            train_params = self.net.parameters()
            self.optimizer = torch.optim.Adam(train_params, lr=cfg.LR, betas=(0.5, 0.999))

        if cfg.MULTIPROCESSING_DISTRIBUTED:
            if cfg.USE_APEX:
                self.net, self.optimizer = apex.amp.initialize(self.net.cuda(), self.optimizer, opt_level=cfg.opt_level)
                self.net = DDP(self.net)

            else:
                self.net = torch.nn.parallel.DistributedDataParallel(self.net.cuda(), device_ids=[cfg.gpu])

        self.optimizers.append(self.optimizer)

        if 'GAN' in self.cfg.LOSS_TYPES:
            self.optimizer_d = torch.optim.SGD(self.discriminator.parameters(), lr=cfg.LR, momentum=0.9,
                                               weight_decay=0.0005)

            if cfg.MULTIPROCESSING_DISTRIBUTED:
                if cfg.USE_APEX:
                    self.discriminator, self.optimizer_d = apex.amp.initialize(self.discriminator.cuda(),
                                                                              self.optimizer_d, opt_level=cfg.opt_level)
                    self.discriminator = DDP(self.discriminator)
                else:
                    self.discriminator = torch.nn.parallel.DistributedDataParallel(self.discriminator.cuda(),
                                                                                  device_ids=[cfg.gpu])

            self.optimizers.append(self.optimizer_d)


    def validate_slide_window(self):

        self.net.eval()
        self.phase = 'test'

        intersection_meter = self.loss_meters['INTERSECTION']
        union_meter = self.loss_meters['UNION']
        target_meter = self.loss_meters['LABEL']

        print('testing with sliding windows...')
        num_images = 0

        # batch = tqdm(self.val_loader)
        # for data in batch:
        for data in self.val_loader:
            self.set_input(data)
            num_images += self.batch_size
            pred = util.slide_cal(model=self.net, image=self.source_modal, crop_size=self.cfg.FINE_SIZE,
                                  prediction_matrix=self.prediction_matrix[0:self.batch_size, :, :, :],
                                  count_crop_matrix=self.count_crop_matrix[0:self.batch_size, :, :, :])

            self.pred = pred.data.max(1)[1]
            intersection, union, label = util.intersectionAndUnionGPU(self.pred, self.label, self.cfg.NUM_CLASSES)
            if self.cfg.MULTIPROCESSING_DISTRIBUTED:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(label)
            intersection, union, label = intersection.cpu().numpy(), union.cpu().numpy(), label.cpu().numpy()

            intersection_meter.update(intersection, self.batch_size)
            union_meter.update(union, self.batch_size)
            target_meter.update(label, self.batch_size)

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        self.loss_meters['VAL_CLS_ACC'].update(allAcc)
        self.loss_meters['VAL_CLS_MEAN_ACC'].update(mAcc)
        self.loss_meters['VAL_CLS_MEAN_IOU'].update(mIoU)

    def validate(self):

        self.phase = 'test'

        # switch to evaluate mode
        self.net.eval()
        self.evaluator.reset()

        intersection_meter = self.loss_meters['INTERSECTION']
        union_meter = self.loss_meters['UNION']
        target_meter = self.loss_meters['LABEL']

        self.pred_index_all = []
        self.target_index_all = []

        with torch.no_grad():

            # batch_index = int(self.val_image_num / cfg.BATCH_SIZE)
            # random_id = random.randint(0, batch_index)

            # batch = tqdm(self.val_loader)
            # for data in batch:
            for i, data in enumerate(self.val_loader):
                self.set_input(data)
                self.data=data#new add
                self._forward(cal_loss=False)

                self._process_fc()

                self.pred = self.result['cls'].data.max(1)[1]
                intersection, union, label = util.intersectionAndUnionGPU(self.pred, self.label,
                                                                          self.cfg.NUM_CLASSES)
                if self.cfg.MULTIPROCESSING_DISTRIBUTED:
                    dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(label)

                intersection, union, _label = intersection.cpu().numpy(), union.cpu().numpy(), label.cpu().numpy()

                intersection_meter.update(intersection, self.batch_size)
                union_meter.update(union, self.batch_size)
                target_meter.update(_label, self.batch_size)

                self.evaluator.add_batch(self.label.cpu().numpy(), self.pred.cpu().numpy())

        # Mean ACC
        self.accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mAcc = np.mean(self.accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        self.loss_meters['VAL_CLS_ACC'].update(allAcc)
        self.loss_meters['VAL_CLS_MEAN_ACC'].update(mAcc)
        # mean_acc = self._cal_mean_acc(self.cfg, self.val_loader)
        # print('mean_acc:', mean_acc)

        if self.cfg.TASK_TYPE == 'segmentation':
            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            mIoU = np.mean(iou_class)
            self.loss_meters['VAL_CLS_MEAN_IOU'].update(mIoU)

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        # # self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        # # self.writer.add_scalar('val/mIoU', mIoU, epoch)
        # # self.writer.add_scalar('val/Acc', Acc, epoch)
        # # self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        # # self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        # print('Validation new:')
        print('Eval: ********')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))

    def _process_fc(self):

        # dist.all_reduce(self.result['cls'])

        _, index = self.result['cls'].data.topk(1, 1, largest=True)

        self.pred_index_all.extend(list(index.cpu().numpy()))
        self.target_index_all.extend(list(self._label.numpy()))

    def _cal_mean_acc(self, cfg, data_loader):

        mean_acc = util.mean_acc(np.array(self.target_index_all), np.array(self.pred_index_all),
                                 cfg.NUM_CLASSES,
                                 data_loader.dataset.classes)
        return mean_acc

    def write_loss(self, phase, global_step=1):

        loss_types = self.cfg.LOSS_TYPES
        task = self.cfg.TASK_TYPE

        label_show = self.label.data.cpu().numpy()
        # if self.phase == 'train':
        #     label_show = self.label.data.cpu().numpy()
        # else:
        #     label_show = np.uint8(self.label.data.cpu())

        source_modal_show = self.source_modal
        target_modal_show = self.target_modal

        if phase == 'train':

            for key, item in self.loss_meters.items():
                if 'TRAIN' in key and item.avg > 0:
                    self.writer.add_scalar(task + '/' + key, item.avg, global_step=global_step)

            self.writer.add_image(task + '/Train_image',
                                  torchvision.utils.make_grid(source_modal_show[:6].clone().cpu().data, 3,
                                                              normalize=True), global_step=global_step)

            # self.writer.add_image(task + '/s0',
            #                       torchvision.utils.make_grid(self.result['s0'][:6].clone().cpu().data, 3,
            #                                                   normalize=True), global_step=global_step)
            # self.writer.add_image(task + '/s0_alphas',
            #                       torchvision.utils.make_grid(self.result['s0_alphas'][:6].clone().cpu().data, 3,
            #                                                   normalize=True), global_step=global_step)
            # self.writer.add_image(task + '/s0_times',
            #                       torchvision.utils.make_grid(self.result['s0_times'][:6].clone().cpu().data, 3,
            #                                                   normalize=True), global_step=global_step)
            # self.writer.add_image(task + '/s1',
            #                       torchvision.utils.make_grid(self.result['s1'][:6].clone().cpu().data, 3,
            #                                                   normalize=True), global_step=global_step)
            # self.writer.add_image(task + '/s2',
            #                       torchvision.utils.make_grid(self.result['s2'][:6].clone().cpu().data, 3,
            #                                                   normalize=True), global_step=global_step)
            # self.writer.add_image(task + '/s3',
            #                       torchvision.utils.make_grid(self.result['s2'][:6].clone().cpu().data, 3,
            #                                                   normalize=True), global_step=global_step)

            if target_modal_show is not None:
                self.writer.add_image(task + '/Train_target',
                                      torchvision.utils.make_grid(target_modal_show[:6].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)

            self.writer.add_scalar(task + '/LR', self.optimizer.param_groups[0]['lr'], global_step=global_step)

            if 'CLS' in loss_types:
                # self.writer.add_scalar(task + '/TRAIN_CLS_LOSS', self.loss_meters['TRAIN_CLS_LOSS'].avg,
                #                        global_step=global_step)
                if 'compl' in self.cfg.MODEL or self.cfg.USE_COMPL_DATA:
                    self.writer.add_scalar(task + '/TRAIN_CLS_LOSS_COMPL', self.loss_meters['TRAIN_CLS_LOSS_COMPL'].avg,
                                           global_step=global_step)
                    self.writer.add_image(task + '/Compl_image',
                                          torchvision.utils.make_grid(self.result['compl_source'][:6].clone().cpu().data, 3,
                                                                      normalize=True), global_step=global_step)
                # self.writer.add_scalar('TRAIN_CLS_ACC', self.loss_meters['TRAIN_CLS_ACC'].avg*100.0,
                #                        global_step=global_step)
                # self.writer.add_scalar('TRAIN_CLS_MEAN_IOU', float(self.train_iou.mean())*100.0,
                #                        global_step=global_step)

            if self.trans and not self.cfg.MULTI_MODAL:

                # if 'SEMANTIC' in self.cfg.LOSS_TYPES:
                #     self.writer.add_scalar(task + '/TRAIN_SEMANTIC_LOSS', self.loss_meters['TRAIN_SEMANTIC_LOSS'].avg,
                #                            global_step=global_step)
                # if 'PIX2PIX' in self.cfg.LOSS_TYPES:
                #     self.writer.add_scalar(task + '/TRAIN_PIX2PIX_LOSS', self.loss_meters['TRAIN_PIX2PIX_LOSS'].avg,
                #                            global_step=global_step)

                self.writer.add_image(task + '/Train_gen',
                                      torchvision.utils.make_grid(self.gen.data[:6].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)
                self.writer.add_image(task + '/Train_image',
                                      torchvision.utils.make_grid(source_modal_show[:6].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)
                # if isinstance(self.target_modal, list):
                #     for i, (gen, target) in enumerate(zip(self.gen, self.target_modal)):
                #         self.writer.add_image('Seg/2_Train_Gen_' + str(self.cfg.FINE_SIZE / pow(2, i)),
                #                               torchvision.utils.make_grid(gen[:6].clone().cpu().data, 3,
                #                                                           normalize=True),
                #                               global_step=global_step)
                #         self.writer.add_image('Seg/3_Train_Target_' + str(self.cfg.FINE_SIZE / pow(2, i)),
                #                               torchvision.utils.make_grid(target[:6].clone().cpu().data, 3,
                #                                                           normalize=True),
                #                               global_step=global_step)
                # else:

            if 'CLS' in loss_types and self.cfg.TASK_TYPE == 'segmentation':
                train_pred = self.result['cls'].data.max(1)[1].cpu().numpy()
                self.writer.add_image(task + '/Train_predicted',
                                      torchvision.utils.make_grid(
                                          torch.from_numpy(util.color_label(train_pred[:6],
                                                                            ignore=self.cfg.IGNORE_LABEL,
                                                                            dataset=self.cfg.DATASET)), 3,
                                          normalize=True, range=(0, 255)), global_step=global_step)
                self.writer.add_image(task + '/Train_label',
                                      torchvision.utils.make_grid(
                                          torch.from_numpy(
                                              util.color_label(label_show[:6], ignore=self.cfg.IGNORE_LABEL,
                                                               dataset=self.cfg.DATASET)), 3, normalize=True,
                                          range=(0, 255)), global_step=global_step)

        elif phase == 'test':

            self.writer.add_image(task + '/Val_image',
                                  torchvision.utils.make_grid(source_modal_show[:6].clone().cpu().data, 3,
                                                              normalize=True), global_step=global_step)
            if self.trans:
                self.writer.add_image(task + '/Val_gen',
                                      torchvision.utils.make_grid(self.gen.data[:24].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)
            # self.writer.add_image('Seg/Val_image',
            #                       torchvision.utils.make_grid(source_modal_show[:6].clone().cpu().data, 3,
            #                                                   normalize=True), global_step=global_step)
            #
            # self.writer.add_image('Seg/Val_predicted',
            #                       torchvision.utils.make_grid(
            #                           torch.from_numpy(util.color_label(self.pred[:6], ignore=self.cfg.IGNORE_LABEL,
            #                                                             dataset=self.cfg.DATASET)), 3,
            #                           normalize=True, range=(0, 255)), global_step=global_step)
            # self.writer.add_image('Seg/Val_label',
            #                       torchvision.utils.make_grid(torch.from_numpy(
            #                           util.color_label(label_show[:6], ignore=self.cfg.IGNORE_LABEL,
            #                                            dataset=self.cfg.DATASET)),
            #                           3, normalize=True, range=(0, 255)),
            #                       global_step=global_step)

            if 'compl' in self.cfg.MODEL or self.cfg.USE_COMPL_DATA:
                self.writer.add_image(task + '/Compl_image',
                                      torchvision.utils.make_grid(self.result['compl_source'][:6].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)

            # for key, loss in self.loss_meters.items():
            #     if 'VAL' in key and loss.avg > 0:
            #         self.writer.add_scalar(task + '/' + key, loss.avg, global_step=global_step)

            self.writer.add_scalar(task + '/VAL_CLS_ACC', self.loss_meters['VAL_CLS_ACC'].val * 100.0,
                                   global_step=global_step)
            self.writer.add_scalar(task + '/VAL_CLS_MEAN_ACC', self.loss_meters['VAL_CLS_MEAN_ACC'].val * 100.0,
                                   global_step=global_step)
            if task == 'segmentation':
                self.writer.add_scalar(task + '/VAL_CLS_MEAN_IOU', self.loss_meters['VAL_CLS_MEAN_IOU'].val * 100.0,
                                       global_step=global_step)

                val_pred = self.result['cls'].data.max(1)[1].cpu().numpy()
                self.writer.add_image(task + '/Val_predicted',
                                      torchvision.utils.make_grid(
                                          torch.from_numpy(util.color_label(val_pred[:6],
                                                                            ignore=self.cfg.IGNORE_LABEL,
                                                                            dataset=self.cfg.DATASET)), 3,
                                          normalize=True, range=(0, 255)), global_step=global_step)
                self.writer.add_image(task + '/Val_label',
                                      torchvision.utils.make_grid(
                                          torch.from_numpy(
                                              util.color_label(label_show[:6], ignore=self.cfg.IGNORE_LABEL,
                                                               dataset=self.cfg.DATASET)), 3, normalize=True,
                                          range=(0, 255)), global_step=global_step)
