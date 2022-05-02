DATA_DIR='chnsenticorp'
CKPT='chnsenticorp_bert'
MAX_SEQ_LEN=300
LABEL_SIZE=2
TEACHER_SURFIX='teacher'

# Run Teacher Model Training and do predict on all samples
python main.py \
  --model 'bert'\
  --loss 'ce'\
  --data_dir $DATA_DIR\
  --nlp_pretrain_model 'bert_base' \
  --ckpt_dir $CKPT\
  --epoch_size 5\
  --batch_size 32\
  --max_seq_len $MAX_SEQ_LEN\
  --lr 1e-5\
  --label_size $LABEL_SIZE\
  --use_gpu \
  --device 6\
  --do_predict\
  --predict_file 'all'

## Split all into train/test/valid
echo 'Split all samples into train/test'
python ./trainsample/converter.py \
  --data_dir "./trainsample/${DATA_DIR}"\
  --input_file "${CKPT}_all"\
  --output_surfix $TEACHER_SURFIX


## Train small model for comparison
echo 'Train small model directly'
python main.py \
  --model 'textcnn'\
  --loss 'ce'\
  --data_dir $DATA_DIR\
  --nlp_pretrain_model 'word2vec_baike' \
  --ckpt_dir "${DATA_DIR}_textcnn" \
  --epoch_size 10\
  --batch_size 64\
  --max_seq_len $MAX_SEQ_LEN\
  --label_size $LABEL_SIZE\
  --lr 1e-3\
  --filter_list '100,100,100'\
  --kernel_size_list '3,4,5'\
  --cnn_activation 'relu'\
  --use_gpu \
  --device 6\
  --train_file "train_${TEACHER_SURFIX}"\
  --valid_file "valid_${TEACHER_SURFIX}"\
  --eval_file "test_${TEACHER_SURFIX}"\
  --clear_model\
  --do_train\
  --do_eval\

## Use Knowledge Distill to train student
echo 'Distill Big Model to Small Model'
python main.py \
  --model 'textcnn'\
  --loss 'ce'\
  --data_dir $DATA_DIR\
  --nlp_pretrain_model 'word2vec_baike'\
  --ckpt_dir "${DATA_DIR}_distill_bert2textcnn"\
  --epoch_size 30\
  --batch_size 64\
  --knowledge_distill\
  --distill_weight 6\
  --distill_loss 'ce'\
  --temperature 5\
  --max_seq_len $MAX_SEQ_LEN\
  --label_size $LABEL_SIZE\
  --lr 1e-4\
  --save_steps 50\
  --filter_list '100,100,100'\
  --kernel_size_list '3,4,5'\
  --cnn_activation 'relu'\
  --use_gpu \
  --device 6\
  --train_file "train_${TEACHER_SURFIX}"\
  --valid_file "valid_${TEACHER_SURFIX}"\
  --eval_file "test_${TEACHER_SURFIX}"\
  --clear_model\
  --do_train\
  --do_eval
