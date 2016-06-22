#!/bin/bash
# Usage: bash split_csv.sh <file.csv> <nb_split> <new_file_suffix>
# Split <file.csv> into <nb_split> files: <new_file_suffix>1.csv, 
# <new_file_suffix>2.csv, ..., <new_file_suffix><nb_split>.csv

file_name=$1
nb_split=$2
new_file_suffix=$3
echo $file_name, $nb_split, $new_file_suffix

number_line=$(wc -l $file_name | cut -d ' ' -f1)
line_split=$(expr $number_line / $nb_split)
echo $line_split
start_line=2
stop_line=$(expr $start_line + $line_split)
for i in `seq 1 $(expr $nb_split - 1)`; do
    sed -n 1,1p $file_name > ${new_file_suffix}${i}.csv
    sed -n "${start_line},${stop_line}p" $file_name >> ${new_file_suffix}${i}.csv
    start_line=$(expr $start_line + $line_split + 1)
    stop_line=$(expr $start_line + $line_split)
done
sed -n 1,1p $file_name > ${new_file_suffix}${nb_split}.csv
sed -n "${start_line},${number_line}p" $file_name >> ${new_file_suffix}${nb_split}.csv
