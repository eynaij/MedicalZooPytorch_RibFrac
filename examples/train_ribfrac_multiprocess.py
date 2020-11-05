# Python libraries
import argparse, os
import torch
import sys

root_dir = os.path.abspath(__file__).split('examples')[0]
sys.path.insert(0, root_dir )

# Lib files
import lib.utils as utils
import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
import lib.train as train
from lib.losses3D import DiceLoss, WeightedCrossEntropyLoss

import apex
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier
import torch.distributed as dist

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
seed = 1777777
torch.manual_seed(seed)


def main():
    args = get_arguments()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled." 
    
    torch.backends.cudnn.benchmark = True 

    utils.reproducibility(args, seed)
    # utils.make_dirs(args.save)
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    training_generator, val_generator, full_volume, affine, dataset = medical_loaders.generate_datasets(args,
                                                                                               path='/data/hejy/MedicalZooPytorch_2cls/datasets')
    model, optimizer = medzoo.create_model(args)
    
    if args.sync_bn:
        model = apex.parallel.convert_syncbn_model(model)

    criterion = DiceLoss(classes=2, skip_index_after=args.classes, weight = torch.tensor([1, 1]).cuda(), sigmoid_normalization=True)
    # criterion = WeightedCrossEntropyLoss()

    if args.cuda:
        model = model.cuda()

    if args.distributed:
        model = DDP(model, delay_allreduce=True)
    # model.restore_checkpoint(args.pretrained)
    trainer = train.Trainer(args, model, criterion, optimizer, train_data_loader=training_generator,
                            valid_data_loader=val_generator, lr_scheduler=None)
    trainer.training()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default="ribfrac")
    parser.add_argument('--dim', nargs="+", type=int, default=(256,256,256))#(128, 128, 48))#(128, 128, 48))#(256,256,256))#(128, 128, 48))#(256,256,256))#(512,512,96))#(256,256,256))#(64,64,48))#(384,384,128)) #(192,192,96))    #   (64,64,48))#(128, 128, 48))  # #   patch_shapes = [(64, 128, 128), (96, 128, 128),(64, 160, 160), (96, 160, 160), (64, 192, 192), (96, 192, 192)]
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--inModalities', type=int, default=1)
    parser.add_argument('--samples_train', type=int, default=1200)
    parser.add_argument('--samples_val', type=int, default=100)
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--threshold', default=0.8, type=float)
    parser.add_argument('--augmentation', default='no', type=str,
                        help='Tensor normalization: options max, mean, global')
    parser.add_argument('--normalization', default='global_mean', type=str,
                        help='Tensor normalization: options max, mean, global')
    parser.add_argument('--loadData', default=False)
    parser.add_argument('--terminal_show_freq', default=1)


    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--split', default=1, type=float, help='Select percentage of training data(default: 0.8)')

    parser.add_argument('--lr', default=1e-2, type=float,
                        help='learning rate (default: 1e-3)')

    parser.add_argument('--lrstep', default=[110, 200], type=float,
                        help='lr decay step ')

    parser.add_argument('--cuda', action='store_true', default=True)

    parser.add_argument('--model', type=str, default='UNET3D',#"SKIPDENSENET3D",#" #'VNET2',#"RESNET3DVAE",#,"RESNETMED3D",#"HIGHRESNET",#'DENSENET1',#'DENSEVOXELNET',#
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--log_dir', type=str,
                        default='../runs/')
    
    parser.add_argument('--model_save_dir_suff', type=str,
                        # default='_512x512x96_thresh0.6_weight1_sample200')
                        # default='_512x512x96_thresh0.6_weight1_sample1200')
                        # default='_256x256x256_thresh0.6_weight1_sample400')
                        # default='_256x256x256_thresh0.6_weight1_sample400_sigmoid')
                        # default='_256x256x256_thresh0.6_weight1_sample1200_sigmoid')
                        default = '_256x256x256_thresh0.8_weight1_sample5400_sigmoid_multipro_6')
                        # default='_256x256x256_thresh0.8_weight1_sample1200_sigmoid')
                        # default='_64x64x48_thresh0.1_weight1_sample400')
                        # default='_128x128x48_thresh0.1_weight1_sample400_softmax_debug')
                        # default='_128x128x48_thresh0.1_weight1_sample400_softmax')
                        # default='_128x128x48_thresh0.1_weight1_sample400_sigmoid_debug')
                        # default='_128x128x48_thresh0.1_weight0.1_sample400_softmax')
                        # default='_128x128x48_thresh0.1_weight1_sample400_wce')
                        # default='_test')
    parser.add_argument('--pretrained',
                        # default='/data/hejy/MedicalZooPytorch/saved_models/UNET3D_checkpoints/UNET3D_64x64x48_0.1_weight0.1/UNET3D_64x64x48_0.1_weight0.1_BEST.pth',
                        # default='/data/hejy/MedicalZooPytorch/saved_models/UNET3D_checkpoints/UNET3D_128x128x48_thresh0.3_weight0.01_sample400_epoch600_test_val+/UNET3D_128x128x48_thresh0.3_weight0.01_sample400_epoch600_test_val+_BEST.pth',
                        # default = '/data/hejy/MedicalZooPytorch_2cls/saved_models/UNET3D_checkpoints/2cls_UNET3D_512x512x96_thresh0.6_weight1_sample200/2cls_UNET3D_512x512x96_thresh0.6_weight1_sample200_BEST.pth',
                        # default='/data/hejy/MedicalZooPytorch_2cls/saved_models/UNET3D_checkpoints/2cls_UNET3D_128x128x48_thresh0.1_weight1_sample400_softmax/2cls_UNET3D_128x128x48_thresh0.1_weight1_sample400_softmax_BEST.pth',
                        default='/data/hejy/MedicalZooPytorch_2cls/saved_models/UNET3D_checkpoints/2cls_UNET3D_256x256x256_thresh0.8_weight1_sample5400_sigmoid_multipro_6_BEST.pth',
                        # default='/data/hejy/MedicalZooPytorch/saved_models/UNET3D_checkpoints/UNET3D_128x128x48_thresh0.3_weight0.01_sample400/UNET3D_128x128x48_thresh0.3_weight0.01_sample400_BEST.pth',
                        # default='/data/hejy/MedicalZooPytorch/saved_models/UNET3D_checkpoints/UNET3D_128x128x48_thresh0.3_sample400_epoch600_test_val+_wce/UNET3D_128x128x48_thresh0.3_sample400_epoch600_test_val+_wce_BEST.pth',
                        type=str, metavar='PATH',
                        help='path to pretrained model')
    parser.add_argument('--distributed', action='store_true', default=True, 
                        help='whether use distributed parallel training')
    parser.add_argument('--sync_bn', action='store_true', default=True, 
                        help='enabling apex sync BN')
    parser.add_argument('--local_rank', default=0, type=int, 
                    help='')
    args = parser.parse_args()

    # args.save = '/data/hejy/MedicalZooPytorch/saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        # utils.datestr(), args.dataset_name)
    
    args.save = '/data/hejy/MedicalZooPytorch_2cls/saved_models/' + args.model + '_checkpoints/' + '2cls_'+ args.model + args.model_save_dir_suff
    return args


if __name__ == '__main__':
    main()
