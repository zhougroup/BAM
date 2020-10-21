ml gcc/7.1.0
ml python3/3.6.1 cuda/9.0 cudnn/7.4.2
SQUAD_DIR=/work/06008/xf993/maverick2/huggingface/SQUAD
ID=0508index0
python3 run_squad.py \
  --seed 1\
  --model_type albert \
  --model_name_or_path albert-base-v2 \
  --do_eval \
  --do_train \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --per_gpu_train_batch_size 3 \  # total batch size should be 12
  --per_gpu_eval_batch_size 3 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --save_steps 5000\
  --output_dir ./results/SQUAD/BE:q!RT/$ID/ \