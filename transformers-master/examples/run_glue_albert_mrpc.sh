ml gcc/7.1.0
ml python3/3.6.1 cuda/9.0 cudnn/7.4.2
GLUE_DIR=/work/06008/xf993/maverick2/huggingface/GLUE/
TASK_NAME=MRPC
ID=0510index0
python3 run_glue.py \
  --seed 1\
  --model_type albert \
  --model_name_or_path albert-base-v2 \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size 4 \
  --gradient_accumulation_steps 2\
  --learning_rate 2e-5 \
  --max_steps 800\
  --warmup_steps 200\
  --doc_stride 128 \
  --num_train_epochs 3.0 \
  --save_steps 799\
  --output_dir ./results/GLUE/$TASK_NAME/ALBERT/$ID/ \
  --do_lower_case \
  --overwrite_output_dir \
  --att_type soft_weibull \
  --label_noise 0.0\
  --k_weibull 10 \
  --kl_anneal_rate 1\
  --att_prior_type constant\
  --alpha_gamma 1e-4\
  --beta_gamma 1e-4\
  --sigma_normal_prior 1e2\
  --sigma_normal_posterior 1e-4\
  --att_kl 1\
  --att_se_hid_size 10\
  --att_se_nonlinear relu
  --k_parameterization orange