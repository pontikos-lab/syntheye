#!/bin/bash
path=/media/pontikos_nas2/NikolasPontikos/UKBB/36741/fundus
echo $path
images=$(ls $path)

t = 0
for i in ${images[@]}; do
  echo $i
  cp $path/$i datasets/fundus_imgs/$i
  t = t + 1
  if [ t == 10 ]
  then
    break
  fi
done
