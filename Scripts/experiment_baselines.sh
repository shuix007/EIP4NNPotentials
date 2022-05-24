#!/bin/bash

declare -a species_list=("Si" "Al")
declare -a gnn_list=("soapnet" "schnet")

split_name=random_2

for species in ${species_list[@]}; do
for gnn in ${gnn_list[@]}; do
    if [ "${species}" == "Al" ]; then
        cut=7
    else
        cut=5
    fi

    python ../finetune.py \
        --cache_dir=../CachedData/${gnn}/${species} \
        --split_dir=../Splits/${species}/ \
        --workspace=../Workspaces/baseline_${gnn}_${species}_cut${cut}_${split_name} \
        --split_name=${split_name} \
        --gnn=${gnn} \
        --cut_r=${cut} \
        --purterb_mode=none &> baseline_${gnn}_${species}_cut${cut}_${split_name}_log.txt
done
done