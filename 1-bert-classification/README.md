#### Bert中文分类任务

- 查看数据
  - 训练数据：https://github.com/shinkeika/nlp_project/blob/master/1-bert-classification/GLUE/glue_data/mydata/train_sentiment.txt
  - 一行一句， 一个tab 后跟一个分类内容，一共17130条
  - 0 negative 1 positive 2 negative

<img src="https://shinkeika.github.io/images/bert/1/1.png" alt="image-20191129162150298" style="zoom:50%;" />

- 看下bert代码需要哪些修改

  - 在 run_classifier.py里看这段代码
  - 其实这个DataProcessor就是处理数据的基类，所有我们自定义的项目都需要继承这个类，并且根据自己的项目的格式重写其中的方法就可以。

  <img src="https://shinkeika.github.io/images/bert/1/2.png" alt="image-20191129162150298" style="zoom:50%;" />

  - 自己的数据处理类

    ```python
    class MyclassificationProcessor(DataProcessor):
        '''
        My classification data processor
        '''
    
        def get_train_examples(self, data_dir):
            '''
            get a collection of 'Input example' for the training set
            :param data_dir: 训练数据的文件
            :return:
            '''
            file_path = os.path.join(data_dir,'train_sentiment.txt')
    
            f = open(file_path, 'r', encoding='UTF-8')
    
            examples = []
    
            for (i, line) in enumerate(f.readlines()):
                guid = "train-%d" % (i)
                line = line.replace('\n','').split('\t')
                text_a = tokenization.convert_to_unicode(str(line[1]))
                label = str(line[2])
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
                )
            return examples
    
        def get_test_examples(self, data_dir):
            '''
            get a collection of 'Input example' for the test set
            :param data_dir: 测试数据的文件
            :return:
            '''
            file_path = os.path.join(data_dir,'test_sentiment.txt')
    
            f = open(file_path, 'r', encoding='UTF-8')
    
            examples = []
    
            for (i, line) in enumerate(f.readlines()):
                guid = "test-%d" % (i)
                line = line.replace('\n','').split('\t')
                text_a = tokenization.convert_to_unicode(str(line[1]))
                label = str(line[2])
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
                )
            return examples
    
        def get_labels(self):
            """Gets the list of labels for this data set."""
            return ['0', '1', '2']
    ```

  - 记得把自己的数据类加入到main函数里<img src="https://shinkeika.github.io/images/bert/1/3.png" alt="image-20191129162150298" style="zoom:50%;" />

  - 那么就可以跑数据了

    `python run_classifier.py \
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
      --output_dir=/Users/shinkeika/learning/nlp_project/1-bert-classification/GLUE/chineseoutput/`

    task_name 是任务名称

    data_dir 是训练数据的位置

    vocab_file 是bert的词表位置

    bert_config_file 是bert的config位置

    max_seq_length 是最大的序列长度，也就是一句话最长的长度

    train_batch_size 电脑性能不好 就选小的

    learning_rate 学习率

    num_train_epochs 训练多少个epochs

    output_dir 输出的位置，包括模型的checkpoint