BERT-BiLSTM-CRF-NER

> 我使用开源项目： https://github.com/macanv/BERT-BiLSTM-CRF-NER

- 安装项目

```
pip install bert-base==0.0.9 -i https://pypi.python.org/simple
```

OR
```angular2html
git clone https://github.com/macanv/BERT-BiLSTM-CRF-NER
cd BERT-BiLSTM-CRF-NER/
python3 setup.py install
```

- 项目目录结构

  - run.py 文件是项目入口

  - data是训练数据的位置
  - chinese_L-12_H-768_A-12是bert预训练的embedding

  <img src="https://shinkeika.github.io/images/bert/2/0.png" style="zoom:50%;" />

- 运行参数设置

```python
-data_dir=/Users/shinkeika/learning/nlp_project/2-BERT-BiLSTM-CRF-NER/data
-output_dir=/Users/shinkeika/learning/nlp_project/2-BERT-BiLSTM-CRF-NER/output
-init_checkpoint=/Users/shinkeika/learning/nlp_project/2-BERT-BiLSTM-CRF-NER/chinese_L-12_H-768_A-12/bert_model.ckpt 
-vocab_file=/Users/shinkeika/learning/nlp_project/2-BERT-BiLSTM-CRF-NER/chinese_L-12_H-768_A-12/vocab.txt 
-bert_config_file=/Users/shinkeika/learning/nlp_project/2-BERT-BiLSTM-CRF-NER/chinese_L-12_H-768_A-12/bert_config.json
```

data_dir 是训练数据的位置

output_dir 模型训练的输出

init_checkpoint bert预训练模型的checkpoint

vocab_file bert预训练的wordembedding的词表位置

bert_config_file bert预先训练配置

- 训练数据结构，标准的NER格式

  <img src="https://shinkeika.github.io/images/bert/2/1.png" alt="image-20191129231439971" style="zoom:50%;" />

  

- 代码解析

  - bert_base/train/train_helper.py 是配置项所在的位置，一般的参数都有默认值，没有特殊需要不用改。

  - train/bert_lstm_ner.py  读取的数据转换为tf_recorder的格式

    <img src="https://shinkeika.github.io/images/bert/2/2.png" alt="image-20191129231120590" style="zoom:50%;" />

  - 转换每个训练样本<img src="https://shinkeika.github.io/images/bert/2/3.png" alt="image-20191129231652689" style="zoom:50%;" />

  - 核心转换方法<img src="https://shinkeika.github.io/images/bert/2/4.png" alt="image-20191129232318905" style="zoom:50%;" />

    最终转换为的数据格式

    input_ids 代表 字的索引

    label_ids 就是该字label里面的ID

    Input_mask  全部都是1 因为 self-attention不管0的事

    segment_ids 只有一句话所以都是0

    <img src="https://shinkeika.github.io/images/bert/2/5.png" alt="image-20191129233019462" style="zoom:50%;" />

  - bert_base/train/models.py 是核心的model构建

    <img src="https://shinkeika.github.io/images/bert/2/6.png" alt="image-20191129234605554" style="zoom:50%;" />

    shape 是[batch size,seq_length, embedding_size] 

    从add_blstm_crf_layer跳进去

    <img src="https://shinkeika.github.io/images/bert/2/7.png" alt="image-20191129234848489" style="zoom:50%;" />

    从project_crf_layer 跳进去<img src="https://shinkeika.github.io/images/bert/2/8.png" alt="image-20191129235124169" style="zoom:50%;" />

    加crf的地方

    <img src="https://shinkeika.github.io/images/bert/2/8.png" alt="image-20191129235407757" style="zoom:50%;" />

- 开始训练

  <img src="https://shinkeika.github.io/images/bert/2/9.png" alt="image-20191129235741768" style="zoom:50%;" />