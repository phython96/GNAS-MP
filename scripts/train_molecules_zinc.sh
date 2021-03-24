
CUDA_VISIBLE_DEVICES=$1 python train/molecules_train.py \
                    --data_name 'ZINC' \
                    --layers 4 \
                    --nodes 3 \
                    --feature_dim 70 \
                    --epochs 150 \
                    --batch_size 128 \
                    --optimizer 'SGD' \
                    --learning_rate 0.025 