import argparse
import os
import sys
root_dir = os.path.abspath(__file__).split('test')[0]
sys.path.insert(0, root_dir )

import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import lib.utils as utils
import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
# from lib.visual3D_temp import non_overlap_padding,test_padding
from lib.losses3D import DiceLoss
from lib.medloaders.miccai_2020_ribfrac import MICCAI2020_RIBFRAC_INFERENCE, MICCAI2020_RIBFRAC_TEST
import matplotlib.pyplot as plt

import nibabel as nib

from lib.visual3D_temp.viz_2d import *

from skimage import morphology, measure, segmentation
from sklearn.cluster import DBSCAN
from collections import Counter

def main():
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    seed = 1777777
    # utils.reproducibility(args, seed)
    # training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args,
    #                                                                                            path='./datasets')
    dataset = MICCAI2020_RIBFRAC_INFERENCE(args, 'val', dataset_path='./datasets', classes=args.classes, dim=args.dim,  #for val
    # dataset = MICCAI2020_RIBFRAC_TEST(args, 'val', dataset_path='./datasets', classes=args.classes, dim=args.dim,     #for test
                                            split_id=0, samples=args.samples_val, load=args.loadData)
    model, optimizer = medzoo.create_model(args)
    print(args.pretrained)
    model.restore_checkpoint(args.pretrained)
    if args.cuda:
        model = model.cuda()
        # full_volume = full_volume.cuda()
        print("Model transferred in GPU.....")
 
    test_net(args, dataset, model,kernel_dim=(32,32,32))
    print('done')
    
def roundup(x, base=32):
    return int(math.ceil(x / base)) * base    

def test_net(args, dataset, model, kernel_dim=(32, 32, 32)):
    # 
    num_imgs = len(dataset)   
    
    public_id_list_all = []
    label_id_list_all = []
    confidence_list_all = []
    label_code_list_all = []
    utils.make_dirs(args.nii_save_path)
    for i in range(num_imgs):    
        print('%d/%d is tested'%(i,num_imgs))
        x, target, img_path, img_affine, img_hdr = dataset.__getitem__(i)
        if args.cuda:
            x = x.unsqueeze(0).cuda()
            # target = target.cuda()
        # 
        # x = full_volume[:-1,...].detach()
        # # target = full_volume[-1,...].unsqueeze(0).detach()
        # target = full_volume[-1,...].detach()
        # # 
        h,w,d = target.shape
        # target_new = torch.zeros(6,h,w,d)
        # target_new[0][target==0] = 1
        # target_new[1][target==1] = 1
        # target_new[2][target==2] = 1
        # target_new[3][target==3] = 1
        # target_new[4][target==4] = 1
        # target_new[5][target==5] = 1
        # target = target_new
       
        modalities, D, H, W = x.shape
        kernel_dim=(512,512,48)  #(256,256,48 )# (512,512,96)#(128, 128, 48) #(192,192,96)       # 512 better
        kc, kh, kw = kernel_dim
        dc, dh, dw = kernel_dim  # stride
        # Pad to multiples of kernel_dim
        a = ((roundup(W, kw) - W) // 2 + W % 2, (roundup(W, kw) - W) // 2,
             (roundup(H, kh) - H) // 2 + H % 2, (roundup(H, kh) - H) // 2,
             (roundup(D, kc) - D) // 2 + D % 2, (roundup(D, kc) - D) // 2)
        x = F.pad(x, a)
        assert x.size(3) % kw == 0
        assert x.size(2) % kh == 0
        assert x.size(1) % kc == 0
        patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        unfold_shape = list(patches.size())
        patches = patches.contiguous().view(-1, modalities, kc, kh, kw)
    
        ## TODO torch stack
        # with torch.no_grad():
        #     output = model.inference(patches)
        number_of_volumes = patches.shape[0]
        predictions = []
    
        for i in range(number_of_volumes):
            input_tensor = patches[i, ...].unsqueeze(0)
            predictions.append(model.inference(input_tensor))
        output = torch.stack(predictions, dim=0).squeeze(1).detach()
        N, Classes, _, _, _ = output.shape
        output_unfold_shape = unfold_shape[1:]
        output_unfold_shape.insert(0, Classes)
        output = output.permute(1,0,2,3,4).contiguous()
        output = output.view(output_unfold_shape)
        output_c = output_unfold_shape[1] * output_unfold_shape[4]
        output_h = output_unfold_shape[2] * output_unfold_shape[5]
        output_w = output_unfold_shape[3] * output_unfold_shape[6]
        output = output.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        output = output.view(-1, output_c, output_h, output_w)
    
        y = output[:, a[4]:output_c - a[5], a[2]:output_h - a[3], a[0]:output_w - a[1]]
        
        # x = x.unsqueeze(0)
        # y = model(x).squeeze(0).cpu().detach()
        # _, h,w,d = y.shape

        
        # y = torch.nn.Softmax(dim=0)(y)
        y = torch.nn.Sigmoid()(y)
        # 
        # y[y<0.5] = 0
       
        index_ = torch.where(y!=0)
        index_fore = torch.where(y[1,:,:,:]>0.05)

        # 
        # confidence_list = y[index_]
        # y_new = torch.zeros(h,w,d)
        # y[index_] = torch.from_numpy(index_[0]).float()
        pred = torch.zeros(h,w,d)
        pred_out = torch.zeros(h,w,d).int()
        # 
        # pred_out = morphology.erosion(pred_out)
        
        conf = torch.zeros(h,w,d)
        # pred[(index_[1], index_[2], index_[3])] = torch.from_numpy(index_[0]).float()
        # conf[(index_[1], index_[2], index_[3])] = y[index_]
        pred[(index_fore[0], index_fore[1], index_fore[2])] = torch.ones_like(index_fore[0]).float()
        pred = torch.from_numpy(morphology.dilation(pred))
        # pred[:130,:,:] = 0. 
        # pred[430:,:,:] = 0. 
        # pred[:,:50,:] = 0. 
        # pred[:,460:,:] = 0. 
        # if pred.shape[2]>300:
        #     pred[:,:,:60] = 0. 
        #     pred[:,:,270:] = 0. 
        index_fore_pred = torch.where(pred!=0)
        conf[(index_fore_pred[0], index_fore_pred[1], index_fore_pred[2])] = y[(torch.ones_like(index_fore_pred[0]),index_fore_pred[0], index_fore_pred[1], index_fore_pred[2])]
        # pred= segmentation.clear_border(pred)
        # 
        # pred = morphology.erosion(pred)
        # for i in range(1, 6):
        #     index_i= np.where(pred==i)
        #     if index_i[0].size == 0:
        #         continue
        #     fea_1 = np.vstack((index_i[0], index_i[1], index_i[2])).T
        #     label_pred = DBSCAN(eps=5).fit_predict(fea_1)
        #     label_counter = Counter(label_pred)
        #     labelId_toRm = []
        #     # labelId_left = []
        #     num_thresh = 10
        #     for label_id, num in label_counter.items():
        #         if num < num_thresh:
        #             labelId_toRm.append(label_id)
        #         # else:
        #             # labelId_left.append(label_id)
        #     if -1 not in labelId_toRm:
        #         labelId_toRm.append(-1) 
        #     index_toRm_list = []
        #     for label_id in labelId_toRm:
        #         index_toRm_list.extend(np.where(label_pred == label_id)[0].tolist())
        #     try:
        #         xyz_toRm =  fea_1[np.array(index_toRm_list)].T
        #     except:
        #         
        #     xyz_toRm_org = (xyz_toRm[0], xyz_toRm[1], xyz_toRm[2])
        #     pred[xyz_toRm_org] = 0.
        # pred = morphology.dilation(morphology.erosion(pred))
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        
        label_id_cnt = 0
        label_code_list = []
        confidence_list = []
        public_id_list = []
        label_id_list = []
        # for i in range(1, 6):
        for i in range(1, 2):
            pred_i = torch.zeros(h,w,d)
            # index_i = np.where(pred == i)
            pred_i[np.where(pred == i)] = 1
            # pred_i = morphology.erosion(pred_i)
            # pred_i = morphology.dilation(pred_i)
            # pred_i = morphology.dilation(pred_i)
            # pred_i = morphology.dilation(pred_i)
            # pred_i = morphology.dilation(morphology.erosion(pred_i))
            labels = measure.label(pred_i, connectivity=3)
            
            confs = []
            # 
            for region in measure.regionprops(labels):
                coords_ = region.coords.T
                coords_org = (coords_[0], coords_[1], coords_[2])
                confs.append(float(torch.max(conf[coords_org])))
            confs.sort()
            if len(confs) > 5:
                for region in measure.regionprops(labels):
                    coords_ = region.coords.T
                    coords_org = (coords_[0], coords_[1], coords_[2])
                    conf_region = float(torch.max(conf[coords_org]))
                    if conf_region < confs[-5]:
                        pred_i[coords_org] = 0
            pred_i = morphology.dilation(pred_i)
            pred_i = morphology.dilation(pred_i)
            pred_i = morphology.dilation(pred_i)

            # 
            # areas = [_.area for _ in measure.regionprops(labels)]
            # areas.sort()
            # if len(areas) > 10:
            #     for region in measure.regionprops(labels):
            #         if region.area < areas[-10]:
            #         # if region.area > areas[0]:
            #             coords_ = region.coords.T
            #             coords_org = (coords_[0], coords_[1], coords_[2])
            #             pred_i[coords_org] = 0
            labels = measure.label(pred_i, connectivity=3)
            # 
            
            for i, region in enumerate(measure.regionprops(labels)):
                # if len(region.coords) < 10000:
                #     continue
                # label_code = int(pred[tuple(region.coords[0].tolist())])
                # label_code = -1 if label_code == 5 else label_code
                label_code=1
                label_code_list.append(label_code)
                label_id_cnt += 1
                label_id_list.append(label_id_cnt)
                coords_ = region.coords.T
                # 
                # ax.scatter(coords_[0], coords_[1], coords_[2], c = np.tile(np.array([label_id_cnt*10]), len(coords_[2])))
                # plt.savefig('scatter.png')
                # 
                coords_org = (coords_[0], coords_[1], coords_[2])
                # confidence = float(torch.sum(conf[coords_org]) / len(coords_org[0]))
                confidence = min(float(torch.max(conf[coords_org]))+0.5, 1)
                pred_out[coords_org] = int(label_id_cnt) #float(label_id_cnt)
                confidence_list.append(confidence)
            # 
        # plt.savefig('scatter.png')
        # 
        label_code_list.insert(0,0)
        label_id_list.insert(0,0)
        confidence_list.insert(0,1)
        pred_cls_num = len(label_code_list)
        public_id = os.path.basename(img_path).split('-')[0]
        public_id_list = [public_id] * pred_cls_num

        public_id_list_all.extend(public_id_list)
        label_id_list_all.extend(label_id_list)
        confidence_list_all.extend(confidence_list)
        label_code_list_all.extend(label_code_list)
        
        test_nii_save_path = os.path.join(args.nii_save_path,public_id+'.nii.gz')
        # nib.Nifti1Image(pred, img_affine).to_filename(test_nii_save_path)
        # pred = pred.numpy().astype(np.int)
        # 
        newLabelImg = nib.Nifti1Image(pred_out.numpy(), img_affine)
        # newLabelImg.set_data_dtype(np.dtype(np.float32))
        # newLabelImg.set_data_dtype(np.dtype(np.int64))
        dimsImgToSave = len(pred_out.shape)
        newZooms = list(img_hdr.get_zooms()[:dimsImgToSave])
        if len(newZooms) < dimsImgToSave : #Eg if original image was 3D, but I need to save a multi-channel image.
            newZooms = newZooms + [1.0]*(dimsImgToSave - len(newZooms))
        newLabelImg.header.set_zooms(newZooms)
        nib.save(newLabelImg, test_nii_save_path)
        #     
    dataframe = pd.DataFrame({'public_id':public_id_list_all, 'label_id':label_id_list_all,'confidence':confidence_list_all, 'label_code':label_code_list_all})
    dataframe.to_csv(args.nii_save_path+"/ribfrac-val-pred.csv",index=False,sep=',')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default="ribfrac")
    parser.add_argument('--dim', nargs="+", type=int, default=(64, 64, 64))
    parser.add_argument('--nEpochs', type=int, default=250)

    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--samples_train', type=int, default=1)
    parser.add_argument('--samples_val', type=int, default=1)
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--inModalities', type=int, default=1)
    parser.add_argument('--fold_id', default='1', type=str, help='Select subject for fold validation')
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='UNET3D',#'DENSENET1',#'DENSEVOXELNET',#
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--loadData', default=False)
    
    # params need to be set for test
    parser.add_argument('--pretrained',
                        # default='/data/hejy/MedicalZooPytorch/saved_models/UNET3D_checkpoints/UNET3D_64x64x48_0.1_weight0.1/UNET3D_64x64x48_0.1_weight0.1_BEST.pth',
                        # default='/data/hejy/MedicalZooPytorch/saved_models/UNET3D_checkpoints/UNET3D_128x128x48_thresh0.3_weight0.01_sample400_epoch600_test_val+/UNET3D_128x128x48_thresh0.3_weight0.01_sample400_epoch600_test_val+_BEST.pth',
                        # default = '/data/hejy/MedicalZooPytorch_2cls/saved_models/UNET3D_checkpoints/2cls_UNET3D_512x512x96_thresh0.6_weight1_sample200/2cls_UNET3D_512x512x96_thresh0.6_weight1_sample200_BEST.pth',
                        # default='/data/hejy/MedicalZooPytorch_2cls/saved_models/UNET3D_checkpoints/2cls_UNET3D_128x128x48_thresh0.1_weight1_sample400_softmax/2cls_UNET3D_128x128x48_thresh0.1_weight1_sample400_softmax_BEST.pth',
                        # default='/data/hejy/MedicalZooPytorch_2cls/saved_models/UNET3D_checkpoints/2cls_UNET3D_512x512x96_thresh0.6_weight1_sample1200/2cls_UNET3D_512x512x96_thresh0.6_weight1_sample1200_BEST_copy.pth',
                        # default='/data/hejy/MedicalZooPytorch_2cls/saved_models/UNET3D_checkpoints/2cls_UNET3D_512x512x96_thresh0.6_weight1_sample1200/2cls_UNET3D_512x512x96_thresh0.6_weight1_sample1200_last_epoch.pth',
                        # default='/data/hejy/MedicalZooPytorch_2cls/saved_models/UNET3D_checkpoints/2cls_UNET3D_128x128x48_thresh0.1_weight1_sample400_wce/2cls_UNET3D_128x128x48_thresh0.1_weight1_sample400_wce_last_epoch.pth',
                        # default='/data/hejy/MedicalZooPytorch_2cls/saved_models/UNET3D_checkpoints/2cls_UNET3D_128x128x48_thresh0.1_weight1_sample400_sigmoid/2cls_UNET3D_128x128x48_thresh0.1_weight1_sample400_sigmoid_last_epoch.pth',
                        # default='/data/hejy/MedicalZooPytorch/saved_models/UNET3D_checkpoints/UNET3D_128x128x48_thresh0.3_weight0.01_sample400/UNET3D_128x128x48_thresh0.3_weight0.01_sample400_BEST.pth',
                        # default='/data/hejy/MedicalZooPytorch/saved_models/UNET3D_checkpoints/UNET3D_128x128x48_thresh0.3_sample400_epoch600_test_val+_wce/UNET3D_128x128x48_thresh0.3_sample400_epoch600_test_val+_wce_BEST.pth',
                        # default='/data/hejy/MedicalZooPytorch_2cls/saved_models/DENSEVOXELNET_checkpoints/2cls_DENSEVOXELNET_128x128x48_thresh0.1_weight1_sample400_softmax/2cls_DENSEVOXELNET_128x128x48_thresh0.1_weight1_sample400_softmax_BEST.pth',
                        # default='/data/hejy/MedicalZooPytorch_2cls/saved_models/UNET3D_checkpoints/2cls_UNET3D_256x256x256_thresh0.6_weight1_sample400/2cls_UNET3D_256x256x256_thresh0.6_weight1_sample400_last_epoch_copy.pth',
                        # default='/data/hejy/MedicalZooPytorch_2cls/saved_models/UNET3D_checkpoints/2cls_UNET3D_256x256x256_thresh0.6_weight1_sample400/2cls_UNET3D_256x256x256_thresh0.6_weight1_sample400_BEST_copy.pth',
                        # default='/data/hejy/MedicalZooPytorch_2cls/saved_models/UNET3D_checkpoints/2cls_UNET3D_128x128x48_thresh0.1_weight1_sample400_sigmoid_augdebug/2cls_UNET3D_128x128x48_thresh0.1_weight1_sample400_sigmoid_augdebug_BEST_copy.pth',
                        # default='/data/hejy/MedicalZooPytorch_2cls/saved_models/UNET3D_checkpoints/2cls_UNET3D_256x256x256_thresh0.6_weight1_sample400_sigmoid_aug_nocrop/2cls_UNET3D_256x256x256_thresh0.6_weight1_sample400_sigmoid_aug_nocrop_last_epoch_copy.pth',
                        # default='/data/hejy/MedicalZooPytorch_2cls/saved_models/DENSENET1_checkpoints/2cls_DENSENET1_128x128x48_thresh0.1_weight1_sample400_sigmoid/2cls_DENSENET1_128x128x48_thresh0.1_weight1_sample400_sigmoid_BEST.pth',
                        # default='/data/hejy/MedicalZooPytorch_2cls/saved_models/UNET3D_checkpoints/2cls_UNET3D_256x256x256_thresh0.8_weight1_sample1200_sigmoid_multipro_ocm_BEST.pth',
                        default = '/data/hejy/MedicalZooPytorch_2cls/saved_models/UNET3D_checkpoints/2cls_UNET3D_256x256x256_thresh0.8_weight1_sample5400_sigmoid_multipro_6_BEST.pth',
                        # default='/data/hejy/MedicalZooPytorch_2cls/saved_models/UNET3D_checkpoints/2cls_UNET3D_256x256x256_thresh0.6_weight1_sample400_sigmoid/2cls_UNET3D_256x256x256_thresh0.6_weight1_sample400_sigmoid_BEST.pth',
                        # default = '/data/hejy/MedicalZooPytorch_2cls/saved_models/UNET3D_checkpoints/2cls_UNET3D_256x256x256_thresh0.6_weight1_sample400_sigmoid_aug_nocrop/2cls_UNET3D_256x256x256_thresh0.6_weight1_sample400_sigmoid_aug_nocrop_BEST.psth',
                        # default = '/data/hejy/MedicalZooPytorch_2cls/saved_models/UNET3D_checkpoints/2cls_UNET3D_128x128x48_thresh0.1_weight1_sample400_sigmoid_aug_nocrop/2cls_UNET3D_128x128x48_thresh0.1_weight1_sample400_sigmoid_aug_nocrop_BEST.pth',
                        type=str, metavar='PATH',
                        help='path to pretrained model')
    parser.add_argument('--nii_save_path', default='/data/hejy/MedicalZooPytorch_2cls/saved_nii')
    # parser.add_argument('--test_path', default="/data/beijing/dataset/MICCAI-RibFrac2020/ribfrac-test-images")
    parser.add_argument('--test_path', default="/data/chelx/MICCAI-RibFrac2020/dataset/ribfrac-val-images")
    # parser.add_argument('--test_path', default="/data/beijing/dataset/MICCAI-RibFrac2020/ribfrac-train-images")
    # parser.add_argument('--test_path', default="/data/hejy/MedicalZooPytorch/datasets/MICCAI_2020_ribfrac/org/ribfrac-val-images")


    args = parser.parse_args()
    args.save = '/data/hejy/MedicalZooPytorch_2cls/inference_checkpoints/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)
    args.tb_log_dir = '../runs/'
    return args


if __name__ == '__main__':
    main()