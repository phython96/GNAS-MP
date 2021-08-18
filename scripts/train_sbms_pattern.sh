DEVICES=$1
GENOTYPE=$2

CUDA_VISIBLE_DEVICES=$DEVICES python train.py \
--task 'node_level' \
--data 'SBM_PATTERN' \
--nb_classes 2 \
--in_dim_V 3 \
--in_dim_E 1 \
--batch 32 \
--edge_feature \
--node_dim 50 \
--edge_dim 50 \
--pos_encode 20 \
--dropout 0.0 \
--batchnorm_op \
--epochs 200 \
--lr 1e-3 \
--weight_decay 0.0 \
--optimizer 'ADAM' \
--load_genotypes $GENOTYPE