#!/bin/bash

# create an image grid using a filepath to image dataset
gen_imgs_path="results/data:faf_dataset_cleaned.csv_classes:classes.txt_trans:512-1-1_mod:cmsggan1-32-512_tr:1000-RAHinge-32-1-0.003-0.003-0.0-0.99/model_ema_state_125/generated_examples"
nImagesPerRow=5
classes=("ABCA4" "USH2A" "RPGR" "BEST1" "PRPH2" "RS1" "TIMP3" "PROML1")
classes=${classes[@]}

function gridMaker {
  gridImages=()
  for c in $classes; do
    images=($(ls $gen_imgs_path/$c))
#    echo ${images[*]::5}
    paths=(${images[*]::$nImagesPerRow})
#    echo ${gridImages[*]}
    for p in ${paths[*]}; do
      gridImages+=($gen_imgs_path/$c/$p)
    done
#    break
  done
#  echo "${gridImages[*]}"
  montage -tile 5x8 ${gridImages[@]::50} $gen_imgs_path/../montage.png
}

gridMaker