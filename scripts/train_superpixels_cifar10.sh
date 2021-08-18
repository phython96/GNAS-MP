DEVICES=$1
GENOTYPE=$2
# no batchnorm + dropout 0.2
CUDA_VISIBLE_DEVICES=$DEVICES python train.py \
--task 'graph_level' \
--data 'CIFAR10' \
--nb_classes 10 \
--in_dim_V 5 \
--batch 64 \
--node_dim 50 \
--pos_encode 0 \
--dropout 0.2 \
--epochs 200 \
--lr 1e-3 \
--weight_decay 0.0 \
--optimizer 'ADAM' \
--load_genotypes $GENOTYPE