#!/bin/bash

# create an image grid using a filepath to image dataset
gen_imgs_path=/home/zchayav/projects/syntheye/synthetic_datasets/all_folds/stylegan2_synthetic_100perclass
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