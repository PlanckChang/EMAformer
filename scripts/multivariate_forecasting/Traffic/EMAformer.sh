export CUDA_VISIBLE_DEVICES=0

model_name=EMAformer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 1024 \
  --output_proj_dropout 0 \
  --train_epochs 50 \
  --n_heads 4 \
  --batch_size 16 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --itr 1 \
  --cycle 168 \
  --lradj type3

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 1 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 1024 \
  --batch_size 16 \
  --train_epochs 50 \
  --output_proj_dropout 0\
  --n_heads 4 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --itr 1 \
  --cycle 168 \
  --lradj type3 

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 1 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 1024 \
  --train_epochs 50 \
  --output_proj_dropout 0 \
  --n_heads 4 \
  --batch_size 16 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --itr 1 \
  --cycle 168 \
  --lradj type3

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 1024 \
  --batch_size 16 \
  --output_proj_dropout 0 \
  --train_epochs 50 \
  --dropout 0.1 \
  --n_heads 4 \
  --learning_rate 0.001\
  --itr 1 \
  --cycle 168 \
  --lradj type3

#  dropout 0.5
#  learning_rate 0.003
#  lradj type1
