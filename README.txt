train.sh
test.sh

lib/medloaders/miccai_2020_ribfrac.py :  dataloader for miccai_2020_ribfrac_competition
lib/medloaders/medical_loader_utils.py
lib/visual3D_temp/BaseWriter.py
lib/medzoo/Unet3D.py                  : dilation, OCM(base)

examples/train_ribfrac_multiprocess.py : train with multiprocess (the version supported currently)
examples/train_ribfrac_aug.py : train with 3d augmentating using batchgenerator(other codes needs to be modified maybe)
examples/train_ribfrac.py

tests/test_ribfrac.py   : test for 2cls
test/test_ribfrac_6cls.py  :test for 6cls