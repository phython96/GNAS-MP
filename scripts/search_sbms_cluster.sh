
CUDA_VISIBLE_DEVICES=$1 python search/sbms_search.py \
                    --data_name 'SBM_CLUSTER' \
                    --batch_size 32 \
                    --layers 4 \
                    --nodes 3 \
                    --feature_dim 96 \
                    --epochs 60 \
                    --save_freq 5 \
                    --save_result './save/CLUSTER_search.txt' \
