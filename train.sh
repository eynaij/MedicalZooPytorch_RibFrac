T=`date +%m%d%H%M`
# CUDA_VISIBLE_DEVICES=7 nohup  python -u examples/train_ribfrac.py > log.3dunet_exp_64_64_48.$T &
# CUDA_VISIBLE_DEVICES=5 nohup  python -u examples/train_ribfrac.py > log.3dunet_exp_128_128_48.$T &
# nohup  python -u examples/train_ribfrac.py > log.3dunet128x128x48_0.1_weight0.01_sample1200.$T &
# python -u examples/train_ribfrac.py 2>&1 | tee log.3dunet__196x196x96_thresh0.1_weight1_sample400.$T 
# python -u examples/train_ribfrac.py 2>&1 | tee log.3dunet_512x512x96_thresh0.6_weight1_sample1200_bg.$T 
# python -u examples/train_ribfrac.py 2>&1 | tee log.2cls_3dunet_512x512x96_thresh0.6_weight1_sample1200.$T 
# python -u examples/train_ribfrac.py 2>&1 | tee log.2cls_3dunet_64x64x48_thresh0.1_weight1_sample400.$T 
# python -u examples/train_ribfrac.py 2>&1 | tee log.2cls_3dunet_128x128x48_thresh0.1_weight1_sample400.$T 
# python -u examples/train_ribfrac.py 2>&1 | tee log.2cls_3dunet_128x128x48_thresh0.1_weight1_sample400_softmax.$T 
# python -u examples/train_ribfrac.py 2>&1 | tee log.2cls_densevoxnet_128x128x48_thresh0.1_weight1_sample400_softmax.$T 
# python -u examples/train_ribfrac.py 2>&1 | tee log.2cls_densevoxnet_128x128x48_thresh0.1_weight0.3_sample400_softmax.$T 
# python -u examples/train_ribfrac.py 2>&1 | tee log.2cls_densevoxnet_128x128x48_thresh0.1_weight0.1_sample400_softmax.$T 
# python -u examples/train_ribfrac.py 2>&1 | tee log.2cls_3dunet_512x512x96_thresh0.6_weight1_sample1200_softmax.$T 
# python -u examples/train_ribfrac.py 2>&1 | tee log.2cls_3dunet_256x256x256_thresh0.6_weight1_sample400_softmax.$T 
# python -u examples/train_ribfrac.py 2>&1 | tee log.2cls_3dunet_128x128x48_thresh0.1_weight1_sample400_sigmoid.$T 
# python -u examples/train_ribfrac.py 2>&1 | tee log.2cls_128x128x48_thresh0.1_weight1_sample400_wce.$T
# python -u examples/train_ribfrac.py 2>&1 | tee log.2cls_3dunet_256x256x256_thresh0.6_weight1_sample400_sigmoid.$T 
# python -u examples/train_ribfrac.py 2>&1 | tee log.2cls_3dunet_256x256x256_thresh0.6_weight1_sample1200_sigmoid.$T 
# python -u examples/train_ribfrac.py 2>&1 | tee log.2cls_3dunet_256x256x256_thresh0.8_weight1_sample1200_sigmoid.$T 
# python -u examples/train_ribfrac.py 2>&1 | tee log.2cls_DENSENET1_128x128x48_thresh0.1_weight1_sample400_sigmoid.$T 
# python -u examples/train_ribfrac.py 2>&1 | tee log.2cls_HIGHRESNET_128x128x48_thresh0.1_weight1_sample400_sigmoid.$T 
# python -u examples/train_ribfrac.py 2>&1 | tee log.2cls_RESNETMED3D_128x128x48_thresh0.1_weight1_sample400_sigmoid.$T 
# python -u examples/train_ribfrac.py 2>&1 | tee log.2cls_VNET2_128x128x48_thresh0.1_weight1_sample400_sigmoid.$T 
# python -u examples/train_ribfrac.py 2>&1 | tee log.2cls_SKIPDENSENET3D_128x128x48_thresh0.1_weight1_sample400_sigmoid.$T 
# python -u examples/train_ribfrac_aug.py 2>&1 | tee log.2cls_SKIPDENSENET3D_256x256x256_thresh0.6_weight1_sample400_sigmoid_aug.$T 
# python -u examples/train_ribfrac_aug.py 2>&1 | tee log.2cls_3dunet_256x256x256_thresh0.6_weight1_sample400_sigmoid_aug_nocrop.$T 
# python -u examples/train_ribfrac_aug.py 2>&1 | tee log.2cls_3dunet_128x128x48_thresh0.1_weight1_sample400_sigmoid_aug_nocrop.$T 
# python -u examples/train_ribfrac.py 2>&1 | tee log.2cls_3dunet_256x256x256_thresh0.8_weight1_sample1200_sigmoid_ocm.$T 
# NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 examples/train_ribfrac_multiprocess.py 2>&1 | tee log.2cls_3dunet_256x256x256_thresh0.8_weight1_sample1200_sigmoid_multipro_ocm.$T 
NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 examples/train_ribfrac_multiprocess.py 2>&1 | tee log.2cls_3dunet_256x256x256_thresh0.8_weight1_sample5400_sigmoid_multipro.$T 




























# python -u examples/train_ribfrac.py