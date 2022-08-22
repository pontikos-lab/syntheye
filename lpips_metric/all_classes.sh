#!/bin/bash
export PYTHONPATH="/home/zchayav/projects/syntheye/PerceptualSimilarity/"
path1=/home/zchayav/projects/syntheye/datasets/eye2gene_new_filepaths/all_baf_valid_50deg_filtered_train_0_edited.csv
path2=/home/zchayav/projects/syntheye/datasets/eye2gene_new_filepaths/all_baf_valid_50deg_filtered_train_0_edited.csv
# path1=/home/zchayav/projects/stylegan2-ada-pytorch/synthetic_datasets/stylegan2_synthetic_-2perclass/generated_examples2.csv
# path2=/home/zchayav/projects/stylegan2-ada-pytorch/synthetic_datasets/stylegan2_synthetic_-2perclass/generated_examples2.csv
while read line; do python ./lpips_metric/compare_pairs.py -p0=$path1 -p1=$path2 -c="$line" -o="./lpips_metric/real_vs_real/lpips_$line.csv" --use_gpu; done < classes.txt
python ./lpips_metric/compare_pairs.py -p0=$path1 -p1=$path2 -c=KCNV2 -o=./lpips_metric/real_vs_real/lpips_KCNV2.csv --use_gpu