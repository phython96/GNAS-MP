
CUDA_VISIBLE_DEVICES=$1 python search/superpixels_search.py \
                    --data_name 'MNIST' \
                    --batch_size 64 \
                    --layers 4 \
                    --nodes 3 \
                    --feature_dim 90 \
                    --epochs 60 \
                    --dropout 0.3 \
                    --save_freq 1 \
                    --save_result './save/MNIST_search.txt' \
