#!/bin/bash
path1=/home/zchayav/projects/syntheye/datasets/eye2gene_new_filepaths/all_baf_valid_50deg_filtered_train_0_edited.csv
path2=$path1
while read line; do python ./lpips/compare_pairs.py -p0=$path1 -p1=$path2 -c=$line -o=lpips_$line.csv --use_gpu; done < classes.txt