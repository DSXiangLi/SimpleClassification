python main.py \
  --model 'textcnn'\
  --loss 'ce'\
  --data_dir 'weibo'\
  --nlp_pretrain_model 'word2vec_baike' \
  --ckpt_dir 'weibo_textcnn_v1' \
  --epoch_size 10\
  --batch_size 64\
  --max_seq_len 1000\
  --lr 1e-3\
  --label_size 2\
  --filter_list '100,100,100'\
  --kernel_size_list '3,4,5'\
  --cnn_activation 'relu'\
  --use_gpu \
  --device 4\
  --enable_cache\
  --clear_cache\
  --clear_model\
  --do_train\
  --do_eval\
  --do_export