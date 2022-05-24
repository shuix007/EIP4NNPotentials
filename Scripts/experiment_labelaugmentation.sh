#!/bin/bash

declare -a species_list=("Si" "Al")
declare -a gnn_list=("soapnet" "schnet")

split_name=random_2
per_atom_energy_cut=0.1

# label augmentation
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
        --workspace=../Workspaces/pretrain_cls_${species}_${gnn}_cut${cut}_${split_name}_Ecut${per_atom_energy_cut} \
        --split_name=${split_name} \
        --gnn=${gnn} \
        --cut_r=${cut} \
        --cls_lambda=1. \
        --reg_lambda=0. \
        --entropy_lambda=0. \
        --cut_per_atom_energy=${per_atom_energy_cut} &> pretrain_cls_${species}_${gnn}_cut${cut}_${split_name}_Ecut${per_atom_energy_cut}_log.txt
done
done