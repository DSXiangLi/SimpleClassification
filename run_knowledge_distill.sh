DATA_DIR='waimai'
CKPT='waimai_bert_v1'
MAX_SEQ_LEN=150
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
  --clear_model\
  --do_train\
  --do_eval\
  --do_predict\

## Split all into train/test/valid
echo 'Shuffle and split all samples into train/test'
python ./trainsample/converter.py \
  --data_dir "./trainsample/${DATA_DIR}"\
  --input_file "${CKPT}_all"\
  --output_surfix $TEACHER_SURFIX

## Use Knowledge Distill to train student
python main.py \
  --model 'fasttext'\
  --loss 'ce'\
  --data_dir $DATA_DIR\
  --nlp_pretrain_model 'word2vec_baike'\
  --ckpt_dir 'distill_bert2fasttext'\
  --epoch_size 5\
  --batch_size 32\
  --knowledge_distill\
  --distill_weight 0.5\
  --distill_loss 'ce'\
  --temperature 2\
  --train_file "train_${TEACHER_SURFIX}"\
  --valid_file "valid_${TEACHER_SURFIX}"\
  --eval_file "test_${TEACHER_SURFIX}"\
  --max_seq_len $MAX_SEQ_LEN\
  --lr 1e-3\
  --label_size $LABEL_SIZE\
  --use_gpu \
  --device 6\
  --do_train\
  --clear_model\
  --do_eval
