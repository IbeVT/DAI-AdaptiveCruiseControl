#!/bin/bash
mkdir ordered
folders=$(ls -d * )
x=0
for folder in $folders; do
        x=$(( RANDOM % 10001 ))
	for i in "$folder/yolo/yolo_dataset/images/train/"*; do
        new_name=$(printf   "$i"_"$x")
        mv "$i" "$new_name"
	mv "$new_name" ./ordered
    done
done

