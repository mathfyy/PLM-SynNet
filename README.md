# Pytorch implementation of 'PLM-SynNet: A Pathology Large Model Synergy Network Based on Multi-instance Learning for Whole Slide Imaging Classification'
<img width="1118" height="690" alt="f2" src="https://github.com/user-attachments/assets/802074ef-47df-4345-b9af-773b1e135058" />



 # Step 1. WSI图像的分割和切块 [Please refer to https://github.com/mahmoodlab/CLAM]
Input: svs格式的WSI数据。
```shell
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --patch --stitch 
```
Output: masks, patches, stitches, process_list_autogen.csv

 # Step 2. patches的特征提取
Input: svs格式的WSI数据, step2得到的patches和process_list_autogen.csv
```shell
python /data1/fengyy/Code/LRENet_debug/dataProcess/extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --batch_size 512 --slide_ext .svs
```

# Step 3. 生成交叉验证所需的列表
Input: 制作csv表, 包含case_id, slide_id, label。
```shell
python create_N_fold_splits_seq.py --task task_gbmlgg --seed 1 --k 5
```
Output: N fold 交叉验证所需的数据分离后的ID表。

# Step 4. 模型训练
main.py

# Step 5. 模型测试
eval.py

## Model
https://pan.baidu.com/s/1aMDC0AYqNwBetpRs-PEy8w
code: 5si4

 ## Requirements
 ### Installation
Please refer to [CLAM](https://github.com/mahmoodlab/CLAM 

## Acknowledgement 
This work was built upon the [CLAM](https://github.com/mahmoodlab/CLAM) and [G-HANet](https://github.com/ZacharyWang-007/G-HANet).

