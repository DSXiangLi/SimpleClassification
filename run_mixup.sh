python main.py \
  --model 'bert'\
  --loss 'sce'\
  --use_mixup\
  --mixup_alpha 0.1\
  --data_dir 'weibo'\
  --nlp_pretrain_model 'bert_base' \
  --ckpt_dir 'weibo_bert_mixup' \
  --epoch_size 5\
  --batch_size 32\
  --max_seq_len 150\
  --lr 1e-5\
  --label_size 2\
  --use_gpu \
  --device 5\
  --enable_cache\
  --clear_model\
  --do_train\
  --do_eval\
  --do_export