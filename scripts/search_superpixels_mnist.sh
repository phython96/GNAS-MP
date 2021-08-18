DEVICES=$1
# no batchnorm + dropout 0.2
CUDA_VISIBLE_DEVICES=$DEVICES python search.py \
--task 'graph_level' \
--data 'MNIST' \
--nb_classes 10 \
--in_dim_V 3 \
--batch 64 \
--epochs 40 \
--node_dim 50 \
--portion 0.5 \
--nb_layers 4 \
--nb_nodes  3 \
--dropout 0.2 \
--nb_workers 0 \
--report_freq 1 \
--search_mode 'train' \
--arch_save 'archs/folder5'
