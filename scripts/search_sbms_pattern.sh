DEVICES=$1

CUDA_VISIBLE_DEVICES=$DEVICES python search.py \
--task 'node_level' \
--data 'SBM_PATTERN' \
--nb_classes 2 \
--in_dim_V 3 \
--in_dim_E 1 \
--data_clip 0.5 \
--batch 8 \
--node_dim 50 \
--edge_dim 50 \
--edge_feature \
--pos_encode 20 \
--nb_layers 1 \
--nb_nodes  4 \
--dropout 0.0 \
--portion 0.5 \
--nb_workers 0 \
--report_freq 1 \
--search_mode 'train' \
--arch_save 'archs/folder5' \
--batchnorm_op 
