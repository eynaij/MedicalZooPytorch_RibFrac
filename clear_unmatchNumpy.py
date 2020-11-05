import numpy as np
import glob
import os
source_path = "/data/hejy/MedicalZooPytorch_2cls/datasets/MICCAI_2020_ribfrac/generated/256/*0.8*0*/"

# dir_path = "/data/hejy/MedicalZooPytorch_2cls/datasets/MICCAI_2020_ribfrac/generated/train_vol_256x256x256_0.8_overall"
dir_path = "/data/hejy/MedicalZooPytorch_2cls/datasets/MICCAI_2020_ribfrac/generated/train_vol_256x256x256_0.8_overall_2"


for i, file_path in enumerate(glob.glob(source_path+'/*.npy')):
    basename = os.path.basename(file_path)
    dst_path = os.path.join(dir_path, basename)
    if os.path.exists(dst_path):
        while os.path.exists(dst_path):
            dst_path = os.path.join(os.path.dirname(dst_path),'copy'+os.path.basename(dst_path))
    os.system('cp -v %s %s' %(file_path, dst_path))

for i, file_path in enumerate(glob.glob(dir_path+'/*.npy')):
    if i % 200 == 0:
        print(i)
    nii_np = np.load(open(file_path,'rb'))
    _, h, w, d = nii_np.shape
    if (h,w,d)!=(256, 256, 256):
        print(nii_np.shape, '\n', file_path)
        os.system('rm -v %s'%file_path)
        
