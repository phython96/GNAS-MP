DEVICES=$1
GENOTYPE=$2
CUDA_VISIBLE_DEVICES=$DEVICES python train.py \
--task 'graph_level' \
--data 'ZINC' \
--in_dim_V 28 \
--batch 128 \
--node_dim 60 \
--dropout 0.0 \
--pos_encode 0 \
--batchnorm_op \
--epochs 200 \
--lr 1e-3 \
--weight_decay 0.0 \
--optimizer 'ADAM' \
--load_genotypes $GENOTYPE
