#!/bin/bash

declare -a species_list=("Si" "Al")
declare -a gnn_list=("soapnet" "schnet")

split_name=random_2

# Pretrain for regression
for species in ${species_list[@]}; do
for gnn in ${gnn_list[@]}; do
    if [ "${species}" == "Al" ]; then
        cut=7
    else
        cut=5
    fi

    python ../pretrain_all.py \
        --species=${species} \
        --cache_dir=../CachedData/${gnn}/${species}/ \
        --split_dir=../Splits/${species}/ \
        --workspace=../Workspaces/pretrain_reg_${species}_${gnn}_cut${cut} \
        --split_name=${split_name} \
        --gnn=${gnn} \
        --cut_r=${cut} \
        --cls_lambda=0. \
        --reg_lambda=1. \
        --entropy_lambda=0. &> pretrain_reg_${species}_${gnn}_cut${cut}_log.txt
done
done