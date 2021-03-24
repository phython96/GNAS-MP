
CUDA_VISIBLE_DEVICES=$1 python search/molecules_search.py \
                    --data_name 'ZINC' \
                    --batch_size 128 \
                    --layers 4 \
                    --nodes 3 \
                    --feature_dim 70 \
                    --epochs 60 \
                    --save_freq 5 \
                    --save_result './save/ZINC_search.txt'\
