#!/bin/bash

bash experiment_baselines.sh && \
bash experiment_pretrain.sh && \
bash experiment_labelaugmentation.sh && \
bash experiment_finetune.sh