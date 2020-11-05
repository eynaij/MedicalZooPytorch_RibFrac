# Python libraries
import argparse
import os

import torch

import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
import lib.train as train
# Lib files
import lib.utils as utils
from lib.losses3D.dice import DiceLoss

from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier
import torch.distributed as dist

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
seed = 1777777
torch.manual_seed(seed)




def main():
    args = get_arguments()
    
    if args.distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."  #1

    torch.backends.cudnn.benchmark = True

    utils.reproducibility(args, seed)
    utils.make_dirs(args.save)

    training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args,
                                                                                               path='.././datasets')
    model, optimizer = medzoo.create_model(args)
    criterion = DiceLoss(classes=11, skip_index_after=args.classes)
    if args.sync_bn:
        model = apex.parallel.convert_syncbn_model(model)

    if args.cuda:
        model = model.cuda()
        print("Model transferred in GPU.....")

    if args.distributed:
        model = DDP(model, delay_allreduce=True)
        
    trainer = train.Trainer(args, model, criterion, optimizer, train_data_loader=training_generator,
                            valid_data_loader=val_generator, lr_scheduler=None)
    print("START TRAINING...")
    trainer.training()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=8)
    parser.add_argument('--dataset_name', type=str, default="mrbrains9")
    parser.add_argument('--dim', nargs="+", type=int, default=(128, 128, 48))
    parser.add_argument('--classes', type=int, default=9)
    parser.add_argument('--nEpochs', type=int, default=200)
    parser.add_argument('--inChannels', type=int, default=3)
    parser.add_argument('--inModalities', type=int, default=3)
    parser.add_argument('--samples_train', type=int, default=10)
    parser.add_argument('--samples_val', type=int, default=10)
    parser.add_argument('--threshold', default=0.1, type=float)
    parser.add_argument('--augmentation', default='no', type=str,
                        help='Tensor normalization: options max, mean, global')
    parser.add_argument('--normalization', default='global_mean', type=str,
                        help='Tensor normalization: options max, mean, global')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--split', default=0.9, type=float, help='Select percentage of training data(default: 0.8)')

    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate (default: 1e-3)')

    parser.add_argument('--cuda', action='store_true', default=False)

    parser.add_argument('--model', type=str, default='UNET3D',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--log_dir', type=str,
                        default='../runs/')
    parser.add_argument('--distributed', action='store_true', default=True, 
                        help='whether use distributed parallel training')
    parser.add_argument('--sync_bn', action='store_true', default=True,
                        help='enabling apex sync BN')
    
    args = parser.parse_args()

    args.save = '/data/hejy/MedicalZooPytorch/saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)
    return args


if __name__ == '__main__':
    main()
