
CUDA_VISIBLE_DEVICES=$1 python train/sbms_train.py \
                    --data_name 'SBM_PATTERN' \
                    --layers 4 \
                    --nodes 3 \
                    --feature_dim 96 \
                    --epochs 150 \
                    --batch_size 32 \
                    --optimizer 'SGD' \
                    --learning_rate 0.025