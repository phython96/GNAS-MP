DEVICES=$1
CUDA_VISIBLE_DEVICES=$DEVICES python search.py \
--task 'node_level' \
--data 'SBM_CLUSTER' \
--nb_classes 6 \
--data_clip 1.0 \
--in_dim_V 7 \
--batch 32 \
--node_dim 70 \
--pos_encode 0 \
--nb_layers 4 \
--nb_nodes  2 \
--dropout 0.2 \
--portion 0.5 \
--search_mode 'train' \
--nb_workers 0 \
--report_freq 1 \
--arch_save 'archs/folder5' \
--batchnorm_op 
