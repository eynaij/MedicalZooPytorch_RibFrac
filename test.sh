python tests/test_ribfrac.py
rm -r saved_nii_older/saved_nii
mv data_4eval/saved_nii saved_nii_older/
cp -r saved_nii data_4eval/
cd RibFrac-Challenge
python ribfrac/evaluation.py --gt_dir '/data/hejy/MedicalZooPytorch_2cls/data_4eval/gt' --pred_dir '/data/hejy/MedicalZooPytorch_2cls/data_4eval/saved_nii'