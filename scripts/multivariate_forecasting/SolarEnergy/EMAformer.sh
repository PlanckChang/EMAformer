export CUDA_VISIBLE_DEVICES=0

model_name=EMAformer

python -u run.py \
  --is_training 1 \
  --use_norm 0 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_96 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 64 \
  --d_ff 128 \
  --n_heads 4 \
  --output_proj_dropout 0 \
  --learning_rate 0.0005 \
  --train_epochs 15 \
  --itr 1 \
  --cycle 144

python -u run.py \
  --is_training 1 \
  --use_norm 0 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_192 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 64 \
  --d_ff 128 \
  --n_heads 4 \
  --train_epochs 15 \
  --learning_rate 0.0005 \
  --output_proj_dropout 0 \
  --itr 1 \
  --cycle 144

python -u run.py \
  --is_training 1 \
  --use_norm 0 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_336 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 4 \
  --train_epochs 15 \
  --learning_rate 0.0005 \
  --output_proj_dropout 0 \
  --itr 1 \
  --cycle 144

python -u run.py \
  --is_training 1 \
  --use_norm 0 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_720 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 64 \
  --d_ff 128 \
  --n_heads 4 \
  --train_epochs 15 \
  --learning_rate 0.0005 \
  --output_proj_dropout 0.5 \
  --itr 1 \
  --cycle 144

