python run_classifier.py \
  --task_name=myclassification \
  --do_train=true \
  --do_eval=true \
  --data_dir=/Users/shinkeika/learning/nlp_project/1-bert-classification/GLUE/glue_data/mydata \
  --vocab_file=/Users/shinkeika/learning/nlp_project/1-bert-classification/GLUE/BERT_BASE_DIR/chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=/Users/shinkeika/learning/nlp_project/1-bert-classification/GLUE/BERT_BASE_DIR/chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=/Users/shinkeika/learning/nlp_project/1-bert-classification/GLUE/BERT_BASE_DIR/chinese_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=70 \
  --train_batch_size=1 \
  --learning_rate=2e-4 \
  --num_train_epochs=1.0 \
  --output_dir=/Users/shinkeika/learning/nlp_project/1-bert-classification/GLUE/chineseoutput/