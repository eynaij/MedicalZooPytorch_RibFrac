import numpy as np
import nibabel as nib
import glob
import os
import h5py

dst_dir = "/data/hejy/datasets/ribfrac/train_h5/"
image_dir = "/data/beijing/dataset/MICCAI-RibFrac2020/ribfrac-train-images/train/"
label_dir = "/data/beijing/dataset/MICCAI-RibFrac2020/ribfrac-train-images/train_label_gt/"

if os.path.exists(dst_dir):
    assert 'hejy' in dst_dir, 'wrong path'
    os.system('rm -r %s' %dst_dir)
os.makedirs(dst_dir)

img_list = []
label_list = []
for i, file_name in enumerate(glob.glob(image_dir+"*/*")):
    print(i, file_name)
    try:
        img = nib.load(file_name).get_fdata()
        file_name = os.path.join(label_dir,file_name.split('/')[-2],file_name.split('/')[-1].replace('image','label')) 
        label = nib.load(file_name).get_fdata()
        assert img.shape == label.shape
        save_name = os.path.join(dst_dir, os.path.basename(file_name).split('-')[0]+'.h5')
        f = h5py.File(save_name,'w')
        f['raw'] = img
        f['label'] = label
        f.close()
        img_list.append(img)
        label_list.append(label)
    except:
        import ipdb;ipdb.set_trace()


imgs= np.array(img_list)
labels = np.array(label_list)

f = h5py.File('ribfrac_train.h5','w')
f['images'] = imgs
f['label'] = labels
f.close()
print('done')

    


