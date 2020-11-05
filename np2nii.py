import numpy as np
import nibabel as nib
import os
import glob

np_dir = '/data/hejy/MedicalZooPytorch_2cls/datasets/MICCAI_2020_ribfrac/generated/train_vol_512x512x96_0.6'
nii_out_dir = '/data/hejy/MedicalZooPytorch_2cls/datasets/MICCAI_2020_ribfrac/generated/train_vol_512x512x96_0.6_nii'

if not os.path.exists(nii_out_dir):
    os.makedirs(nii_out_dir)
for i, np_path in enumerate(glob.glob(np_dir+'/*seg.npy')):
    if i % 100 == 0:
        print(i)
    
    nii_save_name = os.path.join(nii_out_dir, os.path.basename(np_path).split('.')[0] + '.nii.gz')
    np_file = np.load(np_path)
    if len(np_file.shape) ==3:
        np_file = np_file.squeeze(0)
    else:
        _, h, w, d = np_file.shape
        np_file_new = np.zeros((h,w,d))
        index_ = np.where(np_file!=0)
        np_file_new[(index_[1], index_[2], index_[3])] = index_[0]
        np_file = np_file_new
       

    new_image = nib.Nifti1Image(np_file, np.eye(4))
    nib.save(new_image, nii_save_name)