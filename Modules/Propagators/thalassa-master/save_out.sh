#!/bin/bash

echo "Enter directory in which to save output:"
read savedir

if [ ! -d "$savedir" ]; then
  mkdir $savedir
fi

cp ./in/*.txt $savedir/
cp ./out/*.dat $savedir/
cp ./phys/*.txt $savedir/
