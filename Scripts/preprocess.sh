#!/bin/bash

declare -a species_list=("Si" "Al")
declare -a gnn_list=("soapnet" "schnet")

tar xvzf ../RawData/ANI-Al/Al-data.tgz -C ../RawData/ANI-Al/
python ../preprocess_anial.py --input_dir=../RawData/ANI-Al --output_dir=../Data/ANI-Al  

for species in ${species_list[@]}; do
for gnn in ${gnn_list[@]}; do
    if [ "${species}" == "Al" ]; then
        cut=7
    else
        cut=5
    fi

    if [ "${gnn}" == "soapnet" ]; then
        graph_type='soap'
    else
        graph_type='mg'
    fi

    python ../preprocess_graphs.py \
        --data_dir=../Data/ \
        --cache_dir=../CachedData/ \
        --species=${species} \
        --graph_type=${graph_type} \
        --cut_r=${cut} &> preprocess_${gnn}_${species}_cut${cut}_log.txt
done
done