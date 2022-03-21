#!/bin/bash

# create an image grid using a filepath to image dataset
gen_imgs_path="/home/zchayav/projects/syntheye/synthetic_datasets/all_folds/stylegan2_synthetic_100perclass"
nImagesPerRow=6
classes=("ABCA4" "USH2A" "PRPH2" "RPGR" "BEST1" "CHM" "RS1" "RP1" "RHO")
# "RHO" "PRPF31" "MYO7A" "CRB1" "EYS" "CNGB3" "PROML1" "EFEMP1" "TIMP3" 
# "RDH12" "CNGA3" "CACNA1F" "RP2" "GUCY2D" "RPE65" "BBS1" "NR2E3" "MERTK"  
# "CRX" "CERKL" "MTTL1" "OPA1" "PDE6B" "RP1L1" "CYP4V2" "PRPF8" "CDH23" "KCNV2")
classes=${classes[@]}

function gridMaker {
  gridImages=()
  for c in $classes; do
    images=($(ls $gen_imgs_path/$c))
    paths=(${images[*]::$nImagesPerRow})
    for p in ${paths[*]}; do
      gridImages+=($gen_imgs_path/$c/$p)
    done
#    break
  done
#  echo "${gridImages[*]}"
  montage -tile 5x9 -size 5120x4096 ${gridImages[@]::50} $gen_imgs_path/../figure1.png
}

gridMaker