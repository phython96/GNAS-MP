DEVICES=$1
GENOTYPE=$2

CUDA_VISIBLE_DEVICES=$DEVICES python train.py \
--task 'node_level' \
--data 'SBM_CLUSTER' \
--nb_classes 6 \
--in_dim_V 7 \
--in_dim_E 1 \
--edge_feature \
--pos_encode 20 \
--batch 64 \
--node_dim 80 \
--edge_dim 50 \
--dropout 0.0 \
--batchnorm_op \
--epochs 200 \
--lr 1e-3 \
--weight_decay 0.0 \
--optimizer 'ADAM' \
--load_genotypes $GENOTYPE