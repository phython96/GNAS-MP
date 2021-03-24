
CUDA_VISIBLE_DEVICES=$1 python train/superpixels_train.py \
                    --data_name 'CIFAR10' \
                    --layers 4 \
                    --nodes 3 \
                    --feature_dim 90 \
                    --epochs 100 \
                    --batch_size 64 \
                    --optimizer 'SGD' \
                    --learning_rate 0.025 \
                    --dropout 0.3