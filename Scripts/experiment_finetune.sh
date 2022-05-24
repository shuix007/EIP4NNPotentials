#!/bin/bash

declare -a species_list=("Si" "Al")
declare -a gnn_list=("soapnet" "schnet")

split_name=random_2
per_atom_energy_cut=0.1

for species in ${species_list[@]}; do
for gnn in ${gnn_list[@]}; do
    if [ "${species}" == "Al" ]; then
        cut=7
    else
        cut=5
    fi

    # With regression pretraining only
    python ../finetune.py \
        --cache_dir=../CachedData/${gnn}/${species}/ \
        --split_dir=../Splits/${species}/ \
        --pretrained_model_dir=../Workspaces/pretrain_reg_${species}_${gnn}_cut${cut}/best_model.pt \
        --workspace=../Workspaces/${species}_${gnn}_finetune_w_reg_pretrain_cut${cut}_${split_name} \
        --split_name=${split_name} \
        --gnn=${gnn} \
        --num_output_heads=1 \
        --cut_r=${cut} \
        --purterb_mode=none \
        --alpha=0. &> ${species}_${gnn}_finetune_w_reg_pretrain_cut${cut}_${split_name}_log.txt
    
    for alpha in 0.01 0.05 0.1 0.5 1 2 5 10  
    do
        # With augmented labels only
        python ../finetune.py \
            --cache_dir=../CachedData/${gnn}/${species}/ \
            --split_dir=../Splits/${species}/ \
            --purterbed_label_dir=../Workspaces/pretrain_cls_${species}_${gnn}_cut${cut}_${split_name}_Ecut${per_atom_energy_cut}/ \
            --workspace=../Workspaces/${species}_${gnn}_finetune_w_alpha${alpha}_Ecut${per_atom_energy_cut}_cut${cut}_${split_name} \
            --split_name=${split_name} \
            --gnn=${gnn} \
            --num_output_heads=1 \
            --cut_r=${cut} \
            --purterb_mode=ep_pred \
            --alpha=${alpha} &> ${species}_${gnn}_finetune_w_alpha${alpha}_Ecut${per_atom_energy_cut}_cut${cut}_${split_name}_log.txt

        # With both
        python ../finetune.py \
            --cache_dir=../CachedData/${gnn}/${species}/ \
            --split_dir=../Splits/${species}/ \
            --pretrained_model_dir=../Workspaces/pretrain_reg_${species}_${gnn}_cut${cut}/best_model.pt \
            --purterbed_label_dir=../Workspaces/pretrain_cls_${species}_${gnn}_cut${cut}_${split_name}_Ecut${per_atom_energy_cut}/ \
            --workspace=../Workspaces/${species}_${gnn}_finetune_w_reg_alpha${alpha}_Ecut${per_atom_energy_cut}_cut${cut}_${split_name} \
            --split_name=${split_name} \
            --gnn=${gnn} \
            --num_output_heads=1 \
            --cut_r=${cut} \
            --purterb_mode=ep_pred \
            --alpha=${alpha} &> ${species}_${gnn}_finetune_w_reg_alpha${alpha}_Ecut${per_atom_energy_cut}_cut${cut}_${split_name}_log.txt
    done
done
done