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

from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter
from batchgenerators.examples.brats2017.config import brats_preprocessed_folder, num_threads_for_brats_example
from batchgenerators.transforms import Compose
from batchgenerators.utilities.data_splitting import get_split_deterministic
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from lib.medloaders.miccai_2020_ribfrac import MICCAI2020_RIBFRAC, MICCAI2020_RIBFRAC_DataLoader3D

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 1777777
torch.manual_seed(seed)


def main():
    args = get_arguments()
    utils.reproducibility(args, seed)
    # utils.make_dirs(args.save)
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    # training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args,
    training_generator, val_generator, full_volume, affine, dataset = medical_loaders.generate_datasets(args,
                                                                                               path='/data/hejy/MedicalZooPytorch_2cls/datasets')
    model, optimizer = medzoo.create_model(args)

    criterion = DiceLoss(classes=2, skip_index_after=args.classes, weight = torch.tensor([1, 1]).cuda(), sigmoid_normalization=True)
    # criterion = WeightedCrossEntropyLoss()

    if args.cuda:
        model = model.cuda()
    # model.restore_checkpoint(args.pretrained)
    dataloader_train = MICCAI2020_RIBFRAC_DataLoader3D(dataset, args.batchSz, args.dim,  num_threads_in_multithreaded=2)
    tr_transforms = get_train_transform(args.dim)
    training_generator_aug = SingleThreadedAugmenter(dataloader_train, tr_transforms,)
    
    
    trainer = train.Trainer(args, model, criterion, optimizer, train_data_loader=training_generator,
                            valid_data_loader=val_generator, lr_scheduler=None, dataset = dataset, train_data_loader_aug=training_generator_aug)
    trainer.training()


def get_train_transform(patch_size):
    # we now create a list of transforms. These are not necessarily the best transforms to use for BraTS, this is just
    # to showcase some things
    tr_transforms = []

    # the first thing we want to run is the SpatialTransform. It reduces the size of our data to patch_size and thus
    # also reduces the computational cost of all subsequent operations. All subsequent operations do not modify the
    # shape and do not transform spatially, so no border artifacts will be introduced
    # Here we use the new SpatialTransform_2 which uses a new way of parameterizing elastic_deform
    # We use all spatial transformations with a probability of 0.2 per sample. This means that 1 - (1 - 0.1) ** 3 = 27%
    # of samples will be augmented, the rest will just be cropped
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=False,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )

    # now we mirror along all axes
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))

    # brightness transform for 15% of samples
    tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))

    # gamma transform. This is a nonlinear transformation of intensity values
    # (https://en.wikipedia.org/wiki/Gamma_correction)
    tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))
    # we can also invert the image, apply the transform and then invert back
    tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))

    # Gaussian Noise
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))

    # blurring. Some BraTS cases have very blurry modalities. This can simulate more patients with this problem and
    # thus make the model more robust to it
    tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,
                                               p_per_channel=0.5, p_per_sample=0.15))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=4)
    parser.add_argument('--dataset_name', type=str, default="ribfrac")
    parser.add_argument('--dim', nargs="+", type=int, default=(256,256,256))#(128, 128, 48))#(256,256,256))#(256,256,256))#(512,512,96))#(256,256,256))#(64,64,48))#(384,384,128)) #(192,192,96))    #   (64,64,48))#(128, 128, 48))  # #   patch_shapes = [(64, 128, 128), (96, 128, 128),(64, 160, 160), (96, 160, 160), (64, 192, 192), (96, 192, 192)]
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
                        # default='_256x256x256_thresh0.8_weight1_sample1200_sigmoid')
                        default='_256x256x256_thresh0.8_weight1_sample1200_sigmoid_aug_nocrop')
                        # default='_256x256x256_thresh0.6_weight1_sample400_sigmoid_aug_nocrop')
                        # default='_64x64x48_thresh0.1_weight1_sample400')
                        # default='_128x128x48_thresh0.1_weight1_sample400_softmax_debug')
                        # default='_128x128x48_thresh0.1_weight1_sample400_softmax')
                        # default='_128x128x48_thresh0.1_weight1_sample400_sigmoid_augdebug')
                        # default='_128x128x48_thresh0.1_weight0.1_sample400_softmax')
                        # default='_128x128x48_thresh0.1_weight1_sample400_wce')
                        # default='_128x128x48_thresh0.1_weight1_sample400_sigmoid_aug_nocrop')
                        # default='_test')
    parser.add_argument('--pretrained',
                        # default='/data/hejy/MedicalZooPytorch/saved_models/UNET3D_checkpoints/UNET3D_64x64x48_0.1_weight0.1/UNET3D_64x64x48_0.1_weight0.1_BEST.pth',
                        # default='/data/hejy/MedicalZooPytorch/saved_models/UNET3D_checkpoints/UNET3D_128x128x48_thresh0.3_weight0.01_sample400_epoch600_test_val+/UNET3D_128x128x48_thresh0.3_weight0.01_sample400_epoch600_test_val+_BEST.pth',
                        # default = '/data/hejy/MedicalZooPytorch_2cls/saved_models/UNET3D_checkpoints/2cls_UNET3D_512x512x96_thresh0.6_weight1_sample200/2cls_UNET3D_512x512x96_thresh0.6_weight1_sample200_BEST.pth',
                        # default='/data/hejy/MedicalZooPytorch_2cls/saved_models/UNET3D_checkpoints/2cls_UNET3D_128x128x48_thresh0.1_weight1_sample400_softmax/2cls_UNET3D_128x128x48_thresh0.1_weight1_sample400_softmax_BEST.pth',
                        # default='/data/hejy/MedicalZooPytorch/saved_models/UNET3D_checkpoints/UNET3D_128x128x48_thresh0.3_weight0.01_sample400/UNET3D_128x128x48_thresh0.3_weight0.01_sample400_BEST.pth',
                        # default='/data/hejy/MedicalZooPytorch/saved_models/UNET3D_checkpoints/UNET3D_128x128x48_thresh0.3_sample400_epoch600_test_val+_wce/UNET3D_128x128x48_thresh0.3_sample400_epoch600_test_val+_wce_BEST.pth',
                        default='/data/hejy/MedicalZooPytorch_2cls/saved_models/UNET3D_checkpoints/2cls_UNET3D_256x256x256_thresh0.6_weight1_sample400_sigmoid_aug_nocrop/2cls_UNET3D_256x256x256_thresh0.6_weight1_sample400_sigmoid_aug_nocrop_last_epoch_copy.pth',
                        type=str, metavar='PATH',
                        help='path to pretrained model')
    args = parser.parse_args()

    # args.save = '/data/hejy/MedicalZooPytorch/saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        # utils.datestr(), args.dataset_name)
    
    args.save = '/data/hejy/MedicalZooPytorch_2cls/saved_models/' + args.model + '_checkpoints/' + '2cls_'+ args.model + args.model_save_dir_suff
    return args


if __name__ == '__main__':
    main()
