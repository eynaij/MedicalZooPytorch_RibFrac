import glob
import os

import numpy as np
from torch.utils.data import Dataset
import torch

import lib.utils as utils
from lib.medloaders import medical_image_process as img_loader
from lib.medloaders.medical_loader_utils import create_sub_volumes
from lib.medloaders.medical_loader_utils import get_viz_set
from batchgenerators.dataloading.data_loader import DataLoader


class MICCAI2020_RIBFRAC(Dataset):
    def __init__(self, args, mode, dataset_path='../datasets', classes=4, dim=(32, 32, 32), split_id=0, samples=1000,
                 load=False):
        self.mode = mode
        self.root = dataset_path
        self.classes = classes
        dataset_name = 'MICCAI_2020_ribfrac'
        self.training_path = "/data/chelx/MICCAI-RibFrac2020/dataset/ribfrac-train-images"#"/data/beijing/dataset/MICCAI-RibFrac2020/ribfrac-train-images"
        self.dirs = os.listdir(self.training_path)
        self.sample = samples
        self.list = []
        self.full_vol_size = (240,240,48)
        self.threshold = args.threshold
        self.crop_dim = dim
        self.list_reg_t1 = []
        self.labels = []
        self.full_volume = None
        self.save_name = self.root + '/MICCAI_2020_ribfrac/training/classes-' + str(
            classes) + '-list-' + mode + '-samples-' + str(
            samples) + '.txt'
        if load:
            ## load pre-generated data
            self.list = utils.load_list(self.save_name)
            return
        subvol = '_vol_' + str(dim[0]) + 'x' + str(dim[1]) + 'x' + str(dim[2]) + '_' + str(self.threshold) + '_overall'
        # subvol = '_vol_' + str(dim[0]) + 'x' + str(dim[1]) + 'x' + str(dim[2]) + '_' + str(self.threshold)
        # subvol = '_vol_' + str(dim[0]) + 'x' + str(dim[1]) + 'x' + str(dim[2]) + '_' + str(self.threshold) + '_' + str(args.samples_train)

        
        self.sub_vol_path = self.root + '/MICCAI_2020_ribfrac/generated/' + mode + subvol + '/'
        # utils.make_dirs(self.sub_vol_path)
        if not os.path.exists(self.sub_vol_path):
            os.makedirs(self.sub_vol_path)
        if not os.path.exists(os.path.dirname(self.save_name)):
            os.makedirs(os.path.dirname(self.save_name))

        list_reg_t1 = sorted(glob.glob(os.path.join(self.training_path, 'train/*/*.nii.gz')))
        labels = sorted(glob.glob(os.path.join(self.training_path, 'train_label_gt/*/*.nii.gz')))
        
        self.affine = img_loader.load_affine_matrix(list_reg_t1[0])
        
        # split_id = int(split_id)
        # if mode == 'val':
        #     labels = [labels[split_id]]
        #     list_reg_t1 = [list_reg_t1[split_id]]
        # else:
        #     labels.pop(split_id)
        #     list_reg_t1.pop(split_id)
        
        self.list = create_sub_volumes(list_reg_t1, labels,
                                       dataset_name=dataset_name, mode=mode,
                                       samples=samples, full_vol_dim=self.full_vol_size,
                                       crop_size=self.crop_dim, sub_vol_path=self.sub_vol_path,
                                       th_percent=self.threshold)
        
        utils.save_list(self.save_name, self.list)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        t1_path, seg_path = self.list[index]
        # print(t1_path)
        return np.load(t1_path), np.load(seg_path)



class MICCAI2020_RIBFRAC_VAL(Dataset):
    def __init__(self, args, mode, dataset_path='../datasets', classes=4, dim=(32, 32, 32), split_id=0, samples=1000,
                 load=False):
        self.mode = mode
        self.root = dataset_path
        self.classes = classes
        dataset_name = 'MICCAI_2020_ribfrac'
        self.val_path = "/data/chelx/MICCAI-RibFrac2020/dataset/ribfrac-val-images"
        self.dirs = os.listdir(self.val_path)
        self.sample = samples
        self.list = []
        self.full_vol_size = (240,240,48)
        self.threshold = args.threshold
        self.crop_dim = dim
        self.list_reg_t1 = []
        self.labels = []
        self.full_volume = None
        self.save_name = self.root + '/MICCAI_2020_ribfrac/val/classes-' + str(
            classes) + '-list-' + mode + '-samples-' + str(
            samples) + '.txt'
        if load:
            ## load pre-generated data
            self.list = utils.load_list(self.save_name)
            return
        # subvol = '_vol_' + str(dim[0]) + 'x' + str(dim[1]) + 'x' + str(dim[2]) + '_' + str(self.threshold) 
        # subvol = '_vol_' + str(dim[0]) + 'x' + str(dim[1]) + 'x' + str(dim[2]) + '_' + str(self.threshold) + '_' + str(args.samples_train)
        subvol = '_vol_' + str(dim[0]) + 'x' + str(dim[1]) + 'x' + str(dim[2]) + '_' + str(self.threshold) + '_overall'

        self.sub_vol_path = self.root + '/MICCAI_2020_ribfrac/generated/' + mode + subvol + '/'
        # utils.make_dirs(self.sub_vol_path)
        if not os.path.exists(self.sub_vol_path):
            os.makedirs(self.sub_vol_path)
        if not os.path.exists(os.path.dirname(self.save_name)):
            os.makedirs(os.path.dirname(self.save_name))

        list_reg_t1 = sorted(glob.glob(os.path.join(self.val_path, 'val/*.nii.gz')))
        # import ipdb;ipdb.set_trace()
        labels = sorted(glob.glob(os.path.join(self.val_path, 'val_label_gt/*.nii.gz')))
        
        self.affine = img_loader.load_affine_matrix(list_reg_t1[0])
        
        self.list = create_sub_volumes(list_reg_t1, labels,
                                       dataset_name=dataset_name, mode=mode,
                                       samples=samples, full_vol_dim=self.full_vol_size,
                                       crop_size=self.crop_dim, sub_vol_path=self.sub_vol_path,
                                       th_percent=self.threshold)
        
        utils.save_list(self.save_name, self.list)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        t1_path, seg_path = self.list[index]
        return np.load(t1_path), np.load(seg_path)

class MICCAI2020_RIBFRAC_INFERENCE(Dataset):
    def __init__(self, args, mode, dataset_path='../datasets', classes=4, dim=(32, 32, 32), split_id=0, samples=1000,
                 load=False):
        self.mode = mode
        self.root = dataset_path
        self.classes = classes
        dataset_name = 'MICCAI_2020_ribfrac'
        self.val_path = args.test_path#"/data/beijing/dataset/MICCAI-RibFrac2020/ribfrac-val-images"
        self.dirs = os.listdir(self.val_path)
        self.sample = samples
        self.list = []
        self.full_vol_size = (240,240,48)
        # self.threshold = args.threshold
        self.crop_dim = dim
        self.list_reg_t1 = []
        self.labels = []
        self.full_volume = None
        self.save_name = self.root + '/MICCAI_2020_ribfrac/val/classes-' + str(
            classes) + '-list-' + mode + '-samples-' + str(
            samples) + '.txt'
        
        list_reg_t1 = sorted(glob.glob(os.path.join(self.val_path, 'val/*.nii.gz')))
        labels = sorted(glob.glob(os.path.join(self.val_path, 'val_label_gt/*.nii.gz')))
        
        # list_reg_t1 = sorted(glob.glob(os.path.join(self.val_path, 'train/*/*.nii.gz')))
        # labels = sorted(glob.glob(os.path.join(self.val_path, 'train_label_gt/*/*.nii.gz')))
        
        # list_reg_t1 = sorted(glob.glob('/data/hejy/MedicalZooPytorch/datasets/MICCAI_2020_ribfrac/generated/train_vol_128x128x48_0.3_nii/'+ '*.nii.gz'))
        # labels = sorted(glob.glob('/data/hejy/MedicalZooPytorch/datasets/MICCAI_2020_ribfrac/generated/train_vol_128x128x48_0.3_nii/'+ '*.nii.gz'))

        self.affine = img_loader.load_affine_matrix(list_reg_t1[0])
        # self.full_volume = get_viz_set(list_reg_t1, labels, dataset_name=dataset_name)
        self.full_volume = []
       
        for img, label in zip(list_reg_t1, labels):
            img_label_path = []
            img_label_path.append(img)
            img_label_path.append(label)
            self.list.append(tuple(img_label_path))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        img_path, seg_path = self.list[index]
        print(img_path,'\n',seg_path,'\n','----')
        img_np, _, __  = img_loader.load_medical_image_4test(img_path, viz3d=True)
        seg_np, img_affine, img_hdr = img_loader.load_medical_image_4test(seg_path, viz3d=True)
        return img_np, seg_np, img_path, img_affine, img_hdr

class MICCAI2020_RIBFRAC_TEST(Dataset):
    def __init__(self, args, mode, dataset_path='../datasets', classes=4, dim=(32, 32, 32), split_id=0, samples=1000,
                 load=False):
        self.mode = mode
        self.root = dataset_path
        self.classes = classes
        dataset_name = 'MICCAI_2020_ribfrac'
        self.val_path = args.test_path#"/data/beijing/dataset/MICCAI-RibFrac2020/ribfrac-val-images"
        self.dirs = os.listdir(self.val_path)
        self.sample = samples
        self.list = []
        self.full_vol_size = (240,240,48)
        # self.threshold = args.threshold
        self.crop_dim = dim
        self.list_reg_t1 = []
        self.labels = []
        self.full_volume = None
        self.save_name = self.root + '/MICCAI_2020_ribfrac/val/classes-' + str(
            classes) + '-list-' + mode + '-samples-' + str(
            samples) + '.txt'
        
        list_reg_t1 = sorted(glob.glob(os.path.join(self.val_path, '*.nii.gz')))
        # import ipdb;ipdb.set_trace()
        # labels = sorted(glob.glob(os.path.join(self.val_path, 'val_label_gt/*.nii.gz')))
        labels =list_reg_t1

        
        
        # list_reg_t1 = sorted(glob.glob('/data/hejy/MedicalZooPytorch/datasets/MICCAI_2020_ribfrac/generated/train_vol_128x128x48_0.3_nii/'+ '*.nii.gz'))
        # labels = sorted(glob.glob('/data/hejy/MedicalZooPytorch/datasets/MICCAI_2020_ribfrac/generated/train_vol_128x128x48_0.3_nii/'+ '*.nii.gz'))

        self.affine = img_loader.load_affine_matrix(list_reg_t1[0])
        # self.full_volume = get_viz_set(list_reg_t1, labels, dataset_name=dataset_name)
        self.full_volume = []
       
        for img, label in zip(list_reg_t1, labels):
            img_label_path = []
            img_label_path.append(img)
            img_label_path.append(label)
            self.list.append(tuple(img_label_path))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        img_path, seg_path = self.list[index]
        print(img_path,'\n',seg_path,'\n','----')
        img_np, _, __  = img_loader.load_medical_image_4test(img_path, viz3d=True)
        seg_np, img_affine, img_hdr = img_loader.load_medical_image_4test(seg_path, viz3d=True)
        return img_np, seg_np, img_path, img_affine, img_hdr

class MICCAI2020_RIBFRAC_DataLoader3D(DataLoader):
    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded, seed_for_shuffle=1234,
                 return_incomplete=False, shuffle=True, infinite=True):
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)
        self.patch_size = patch_size
        self.num_modalities = 1
        self.indices = list(range(len(data)))
    
    def generate_train_batch(self):
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]

        data = np.zeros((self.batch_size, self.num_modalities, *self.patch_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, 2, *self.patch_size), dtype=np.float32)

        for i, j in enumerate(patients_for_batch):
            data[i] = j[0]
            seg[i] = j[1]
        return {'data': data, 'seg':seg}