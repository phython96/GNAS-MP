
CUDA_VISIBLE_DEVICES=$1 python search/sbms_search.py \
                    --data_name 'SBM_PATTERN' \
                    --batch_size 32 \
                    --layers 4 \
                    --nodes 3 \
                    --feature_dim 96 \
                    --epochs 60 \
                    --save_freq 5 \
                    --save_result './save/PATTERN_search.txt' \
