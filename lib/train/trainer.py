import numpy as np
import torch

from lib.utils.general import prepare_input
from lib.visual3D_temp.BaseWriter import TensorboardWriter

from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter
from batchgenerators.examples.brats2017.config import brats_preprocessed_folder, num_threads_for_brats_example
from batchgenerators.transforms import Compose
from batchgenerators.utilities.data_splitting import get_split_deterministic
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import argparse, os, shutil


class Trainer:
    """
    Trainer class
    """

    def __init__(self, args, model, criterion, optimizer, train_data_loader,
                #  valid_data_loader=None, lr_scheduler=None):
                 valid_data_loader=None, lr_scheduler=None,dataset=None, train_data_loader_aug=None ):
        self.dataset = dataset
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_data_loader = train_data_loader
        self.train_data_loader_aug = train_data_loader_aug
        # epoch-based training
        self.len_epoch = len(self.train_data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size))
        self.writer = TensorboardWriter(args)

        self.save_frequency = 10
        self.terminal_show_freq = self.args.terminal_show_freq
        self.start_epoch = 1

    def adjust_learning_rate(self, optimizer, gamma, step):
        lr = self.args.lr * (gamma ** (step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def training(self):
        step_index = 0
        best_loss = 1000000
        for epoch in range(self.start_epoch, self.args.nEpochs):
            # import ipdb;ipdb.set_trace()
            if epoch in self.args.lrstep:
                step_index += 1
                self.adjust_learning_rate(self.optimizer, 0.1, step_index)
                
            if self.train_data_loader_aug is None:
                self.train_epoch(epoch)
            else:
                self.train_epoch_aug(epoch)
            
            if self.do_validation:
                self.validate_epoch(epoch)

            val_loss = self.writer.data['val']['loss'] / self.writer.data['val']['count']

            if self.args.save is not None and ((epoch + 1) % self.save_frequency):
                # if self.args.local_rank == 0:
                #     self.model.save_checkpoint(self.args.save,
                #                                epoch, val_loss,
                #                                optimizer=self.optimizer)
                is_best = best_loss > val_loss
                best_prec1 = min(best_loss, val_loss)
                if self.args.local_rank == 0:
                    print('Save model to disk')
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                    }, is_best, filename='{}.pth'.format(self.args.save))
            
            

            if self.args.local_rank == 0:
                self.writer.write_end_of_epoch(epoch)

            self.writer.reset('train')
            self.writer.reset('val')

    def train_epoch_aug(self, epoch):
        self.model.train()
        
        # args = get_arguments()
        # train_dataset = MICCAI2020_RIBFRAC(args, 'train', dataset_path='../datasets', classes=args.classes, dim=args.dim,
        #                                       split_id=0, samples=args.samples_train, load=args.loadData)
        # patch_size = (128, 128, 48)
        # batch_size = 2 
        # num_threads_for_brats_example = 2
        
        # dataloader_train = MICCAI2020_RIBFRAC_DataLoader3D(self.dataset, batch_size, patch_size, num_threads_for_brats_example)

        # tr_transforms = get_train_transform(patch_size)
        # tr_gen = SingleThreadedAugmenter(dataloader_train, tr_transforms,) #num_processes=num_threads_for_brats_example,
                                   # num_cached_per_queue=3,
                                  #  seeds=None, 
                                  #  pin_memory=False)
        # tr_gen.restart()
        # _ = next(tr_gen)
        # import ipdb;ipdb.set_trace()
        # for batch_idx, input_tuple in enumerate(self.train_data_loader):
        for batch_idx, data_seg_dict in enumerate(self.train_data_loader_aug):
        # for batch_idx, data_seg_dict in enumerate(tr_gen):
            if batch_idx / self.len_epoch == 1:
                break
            input_tuple = [torch.from_numpy(data_seg_dict['data']),torch.from_numpy(data_seg_dict['seg'])]
            # input_tuple = [data_seg_dict['data'],data_seg_dict['seg']]
            self.optimizer.zero_grad()
            # import ipdb;ipdb.set_trace()
            input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
            input_tensor.requires_grad = True
            output = self.model(input_tensor)
            # import ipdb;ipdb.set_trace()
            # loss_dice, per_ch_score = self.criterion(output[0], target)
            loss_dice, per_ch_score = self.criterion(output, target)
            # loss_dice = self.criterion(output, target)
            # print('epoch ',epoch, 'loss: ',loss_dice.item())
            loss_dice.backward()
            self.optimizer.step()
            # import ipdb;ipdb.set_trace()
            if self.args.local_rank == 0:
                self.writer.update_scores(batch_idx, loss_dice.item(), per_ch_score, 'train',
                                      epoch * self.len_epoch + batch_idx)

            if (batch_idx + 1) % self.terminal_show_freq == 0:
                partial_epoch = epoch + batch_idx / self.len_epoch - 1
                if self.args.local_rank == 0:
                    self.writer.display_terminal(partial_epoch, epoch, 'train')
        if self.args.local_rank == 0:
            self.writer.display_terminal(self.len_epoch, epoch, mode='train', summary=True)

    def train_epoch(self, epoch):
        self.model.train()

        for batch_idx, input_tuple in enumerate(self.train_data_loader):

            self.optimizer.zero_grad()
            # import ipdb;ipdb.set_trace()
            input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
            input_tensor.requires_grad = True
            output = self.model(input_tensor)
            # import ipdb;ipdb.set_trace()
            # loss_dice, per_ch_score = self.criterion(output[0], target)
            loss_dice, per_ch_score = self.criterion(output, target)
            # loss_dice = self.criterion(output, target)
            # print('epoch ',epoch, 'loss: ',loss_dice.item())
            loss_dice.backward()
            self.optimizer.step()
            if self.args.local_rank == 0:
                self.writer.update_scores(batch_idx, loss_dice.item(), per_ch_score, 'train',
                                          epoch * self.len_epoch + batch_idx)

            if (batch_idx + 1) % self.terminal_show_freq == 0:
                partial_epoch = epoch + batch_idx / self.len_epoch - 1
                if self.args.local_rank == 0:
                    self.writer.display_terminal(partial_epoch, epoch, 'train')
        if self.args.local_rank == 0:
            self.writer.display_terminal(self.len_epoch, epoch, mode='train', summary=True)

    def validate_epoch(self, epoch):
        self.model.eval()

        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
            with torch.no_grad():
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                input_tensor.requires_grad = False

                output = self.model(input_tensor)
                # loss, per_ch_score = self.criterion(output[0], target)
                loss, per_ch_score = self.criterion(output, target)
                # loss_dice = self.criterion(output, target)
                # print('epoch ',epoch , 'loss: ',loss_dice.item())
                if self.args.local_rank == 0:
                    self.writer.update_scores(batch_idx, loss.item(), per_ch_score, 'val',
                                              epoch * self.len_epoch + batch_idx)
        if self.args.local_rank == 0:
            self.writer.display_terminal(len(self.valid_data_loader), epoch, mode='val', summary=True)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        best_name = os.path.splitext(filename)[0] + '_BEST.pth'
        shutil.copyfile(filename, best_name)