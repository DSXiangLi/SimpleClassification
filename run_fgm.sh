python main.py \
  --model 'bert'\
  --loss 'ce'\
  --use_fgm\
  --epsilon 5.0\
  --data_dir 'chnsenticorp'\
  --nlp_pretrain_model 'bert_base' \
  --ckpt_dir 'chnsenticorp_fgm' \
  --epoch_size 5\
  --batch_size 8\
  --max_seq_len 200\
  --lr 5e-6\
  --early_stop_ratio 0.5\
  --use_gpu\
  --device 0\
  --do_train\
  --do_eval