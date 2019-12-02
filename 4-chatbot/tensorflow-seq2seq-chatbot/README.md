#### 使用开源项目tensorflow/nmt


```python
# 下载项目
!git clone https://github.com/tensorflow/nmt/
!ls
```

    fatal: destination path 'nmt' already exists and is not an empty directory.
    README.md             [1m[36mnmt[m[m                   seq2seq-chatbot.ipynb


**我们下载小黄鸡语料，并对它做一个处理，使得它符合seq2seq模型的输入格式**


```python
!wget https://github.com/candlewill/Dialog_Corpus/raw/master/xiaohuangji50w_nofenci.conv.zip
!unzip xiaohuangji50w_nofenci.conv.zip
```

    --2019-12-02 10:29:01--  https://github.com/candlewill/Dialog_Corpus/raw/master/xiaohuangji50w_nofenci.conv.zip
    正在解析主机 github.com (github.com)... 192.30.253.112
    正在连接 github.com (github.com)|192.30.253.112|:443... 已连接。
    已发出 HTTP 请求，正在等待回应... 302 Found
    位置：https://raw.githubusercontent.com/candlewill/Dialog_Corpus/master/xiaohuangji50w_nofenci.conv.zip [跟随至新的 URL]
    --2019-12-02 10:29:02--  https://raw.githubusercontent.com/candlewill/Dialog_Corpus/master/xiaohuangji50w_nofenci.conv.zip
    正在解析主机 raw.githubusercontent.com (raw.githubusercontent.com)... 151.101.76.133
    正在连接 raw.githubusercontent.com (raw.githubusercontent.com)|151.101.76.133|:443... 已连接。
    已发出 HTTP 请求，正在等待回应... 200 OK
    长度：10232473 (9.8M) [application/zip]
    正在保存至: “xiaohuangji50w_nofenci.conv.zip”
    
    xiaohuangji50w_nofe 100%[===================>]   9.76M  5.80MB/s  用时 1.7s      
    
    2019-12-02 10:29:05 (5.80 MB/s) - 已保存 “xiaohuangji50w_nofenci.conv.zip” [10232473/10232473])
    
    Archive:  xiaohuangji50w_nofenci.conv.zip
      inflating: xiaohuangji50w_nofenci.conv  



```python
# 把第一行的E和换行换后的第二行换成Q M
!perl -pi.bak -e 's/(E\s)/\1Q /g' xiaohuangji50w_nofenci.conv
# 把第二行的(Q, M)换成 Q
!perl -pi.bak -e 's/(Q M)/Q/g' xiaohuangji50w_nofenci.conv
# 第三行的M换成 A
!perl -pi.bak -e 's/(M )/A /g' xiaohuangji50w_nofenci.conv
```


```python
!head -30 xiaohuangji50w_nofenci.conv
```

    E
    Q 呵呵
    A 是王若猫的。
    E
    Q 不是
    A 那是什么？
    E
    Q 怎么了
    A 我很难过，安慰我~
    E
    Q 开心点哈,一切都会好起来
    A 嗯 会的
    E
    Q 我还喜欢她,怎么办
    A 我帮你告诉她？发短信还是打电话？
    E
    Q 短信
    A 嗯嗯。我也相信
    E
    Q 你知道谁么
    A 肯定不是我，是阮德培
    E
    Q 许兵是谁
    A 吴院四班小帅哥
    E
    Q 这么假
    A 三鹿奶粉也假，不一样的卖啊
    E
    Q 许兵是傻逼
    A 被你发现了。



```python
text = open('xiaohuangji50w_nofenci.conv').read().split('E\n')
text[1]
```




    'Q 呵呵\nA 是王若猫的。\n'



**分词**


```python
import jieba
def split_conv(in_f, out_q, out_a):
    out_question = open(out_q, 'w')
    out_answer = open(out_a, 'w')
    text = open(in_f).read().split('E\n')
    for pair in text:
        # 句子长度太短的对话，就过滤掉，跳过
        if len(pair) <= 4:
            continue
        # 切分问题和回答
        contents = pair.split('\n')
        out_question.write(' '.join(jieba.lcut(contents[0].strip('Q '))) + '\n')
        out_answer.write(' '.join(jieba.lcut(contents[1].strip('A '))) + '\n')
    out_question.close()
    out_answer.close()
```


```python
in_f = 'xiaohuangji50w_nofenci.conv'
out_q = 'question.file'
out_a = 'answer.file'
split_conv(in_f, out_q, out_a)
```

    Building prefix dict from the default dictionary ...
    Loading model from cache /var/folders/g_/rv2sg65j1_g2znz05v_snmth0000gn/T/jieba.cache
    Loading model cost 0.652 seconds.
    Prefix dict has been built succesfully.


**查看question的前10行**


```python
!head -10 question.file
```

    呵呵
    不是
    怎么 了
    开心 点哈 , 一切 都 会 好 起来
    我 还 喜欢 她 , 怎么办
    短信
    你 知道 谁 么
    许兵 是 谁
    这么 假
    许兵 是 傻 逼


**查看答案的前10行**


```python
!head -10 answer.file
```

    是 王若 猫 的 。
    那 是 什么 ？
    我 很 难过 ， 安慰 我 ~
    嗯   会 的
    我 帮 你 告诉 她 ？ 发短信 还是 打电话 ？
    嗯 嗯 。 我 也 相信
    肯定 不是 我 ， 是 阮德培
    吴院 四班 小帅哥
    三鹿 奶粉 也 假 ， 不 一样 的 卖 啊
    被 你 发现 了 。


**查看问题一共有多少行**


```python
!wc -l question.file
```

      454131 question.file


**查看答案一共有多少行**


```python
!wc -l answer.file
```

      454131 answer.file



```python
import re
def get_vocab(in_f, out_f):
    vocab_dic = {}
    for line in open(in_f, encoding='utf-8'):
        words = line.strip().split(' ')
        for word in words:
            # 保留汉字内容
            if not re.match(r'[\u4e00-\u9fa5]+', word):
                continue
            try:
                vocab_dic[word] += 1
            except:
                vocab_dic[word] = 1
    out = open(out_f, 'w', encoding='utf-8')
    out.write("<unk>\n<s>\n</s>\n")
    vocab = sorted(vocab_dic.items(), key=lambda x:x[1],reverse=True)
    
    for word in [x[0] for x in vocab[:800000]]:
        out.write(word)
        out.write('\n')
    out.close()
```

**切分训练，验证，测试集**


```python
!mkdir data
# 前300000作为训练集
!head -300000 question.file > data/train.input
!head -300000 answer.file > data/train.output
# 后80000作为验证集
!head -380000 question.file | tail -80000 > data/val.input
!head -380000 question.file | tail -80000 > data/val.output
# 最后750000作为测试集
!tail -75000 question.file > data/test.input
!tail -75000 answer.file > data/test.output
```


```python
in_file = 'question.file'
out_file = './data/vocab.input'
get_vocab(in_file, out_file)
```


```python
in_file = 'answer.file'
out_file = './data/vocab.output'
get_vocab(in_file, out_file)
```


```python
!mkdir data/nmt_attention_model
```

**参考 [nmt的超参数](https://luozhouyang.github.io/tensorflow_nmt_hparams/)**


```python
# !python3 -m nmt.nmt \  
#     --attention=scaled_luong \  # 使用attention 的方式
#     --src=input --tgt=output \  # 源的后座
#     --vocab_prefix=./data/vocab  \  # vocab 的前缀
#     --train_prefix=./data/train \  # 训练数据的前缀
#     --dev_prefix=./data/val  \  # 验证集的前缀
#     --test_prefix=./data/test \  # 训练集的前缀
#     --out_dir=/tmp/nmt_attention_model \  # 输出的文件夹
#     --num_train_steps=12000 \  # 迭代的步数
#     --steps_per_stats=1 \  # 多少步输出一次状态
#     --num_layers=2 \  # 每个cell有多少层
#     --num_units=128 \  # 有多少个神经元
#     --dropout=0.2 \  # dropout的比率
#     --metrics=bleu  # 评估指标
```


```python
!python -m nmt.nmt.nmt \
    --attention=scaled_luong \
    --src=input --tgt=output \
    --vocab_prefix=./data/vocab  \
    --train_prefix=./data/train \
    --dev_prefix=./data/val  \
    --test_prefix=./data/test \
    --out_dir=/tmp/nmt_attention_model \
    --num_train_steps=12000 \
    --steps_per_stats=1 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu 
```

    /Users/shinkeika/anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:526: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /Users/shinkeika/anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:527: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /Users/shinkeika/anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:528: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /Users/shinkeika/anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:529: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /Users/shinkeika/anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:530: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /Users/shinkeika/anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:535: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])
    
    WARNING: The TensorFlow contrib module will not be included in TensorFlow 2.0.
    For more information, please see:
      * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
      * https://github.com/tensorflow/addons
    If you depend on functionality not listed there, please file an issue.
    
    # Job id 0
    2019-12-02 12:01:14.867781: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
    # Devices visible to TensorFlow: [_DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 268435456, 12028887416896540267)]
    # Creating output directory /tmp/nmt_attention_model ...
    # Vocab file ./data/vocab.input exists
    # Vocab file ./data/vocab.output exists
      saving hparams to /tmp/nmt_attention_model/hparams
      saving hparams to /tmp/nmt_attention_model/best_bleu/hparams
      attention=scaled_luong
      attention_architecture=standard
      avg_ckpts=False
      batch_size=128
      beam_width=0
      best_bleu=0
      best_bleu_dir=/tmp/nmt_attention_model/best_bleu
      check_special_token=True
      colocate_gradients_with_ops=True
      coverage_penalty_weight=0.0
      decay_scheme=
      dev_prefix=./data/val
      dropout=0.2
      embed_prefix=None
      encoder_type=uni
      eos=</s>
      epoch_step=0
      forget_bias=1.0
      infer_batch_size=32
      infer_mode=greedy
      init_op=uniform
      init_weight=0.1
      language_model=False
      learning_rate=1.0
      length_penalty_weight=0.0
      log_device_placement=False
      max_gradient_norm=5.0
      max_train=0
      metrics=['bleu']
      num_buckets=5
      num_dec_emb_partitions=0
      num_decoder_layers=2
      num_decoder_residual_layers=0
      num_embeddings_partitions=0
      num_enc_emb_partitions=0
      num_encoder_layers=2
      num_encoder_residual_layers=0
      num_gpus=1
      num_inter_threads=0
      num_intra_threads=0
      num_keep_ckpts=5
      num_sampled_softmax=0
      num_train_steps=12000
      num_translations_per_input=1
      num_units=128
      optimizer=sgd
      out_dir=/tmp/nmt_attention_model
      output_attention=True
      override_loaded_hparams=False
      pass_hidden_state=True
      random_seed=None
      residual=False
      sampling_temperature=0.0
      share_vocab=False
      sos=<s>
      src=input
      src_embed_file=
      src_max_len=50
      src_max_len_infer=None
      src_vocab_file=./data/vocab.input
      src_vocab_size=56491
      steps_per_external_eval=None
      steps_per_stats=1
      subword_option=
      test_prefix=./data/test
      tgt=output
      tgt_embed_file=
      tgt_max_len=50
      tgt_max_len_infer=None
      tgt_vocab_file=./data/vocab.output
      tgt_vocab_size=50041
      time_major=True
      train_prefix=./data/train
      unit_type=lstm
      use_char_encode=False
      vocab_prefix=./data/vocab
      warmup_scheme=t2t
      warmup_steps=0
    WARNING:tensorflow:From /Users/shinkeika/learning/nlp_project/4-chatbot/tensorflow-seq2seq-chatbot/nmt/nmt/utils/iterator_utils.py:129: DatasetV1.shard (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `dataset.apply(tf.data.experimental.filter_for_shard(...))`.
    WARNING:tensorflow:From /Users/shinkeika/learning/nlp_project/4-chatbot/tensorflow-seq2seq-chatbot/nmt/nmt/utils/iterator_utils.py:235: group_by_window (from tensorflow.contrib.data.python.ops.grouping) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.experimental.group_by_window(...)`.
    WARNING:tensorflow:From /Users/shinkeika/learning/nlp_project/4-chatbot/tensorflow-seq2seq-chatbot/nmt/nmt/utils/iterator_utils.py:228: to_int64 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    WARNING:tensorflow:From /Users/shinkeika/anaconda3/lib/python3.7/site-packages/tensorflow/python/data/ops/dataset_ops.py:1419: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    # Creating train graph ...
    # Build a basic encoder
      num_layers = 2, num_residual_layers=0
      cell 0  LSTM, forget_bias=1WARNING:tensorflow:From /Users/shinkeika/learning/nlp_project/4-chatbot/tensorflow-seq2seq-chatbot/nmt/nmt/model_helper.py:402: BasicLSTMCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This class is equivalent as tf.keras.layers.LSTMCell, and will be replaced by that in Tensorflow 2.0.
      DropoutWrapper, dropout=0.2   DeviceWrapper, device=/gpu:0
      cell 1  LSTM, forget_bias=1  DropoutWrapper, dropout=0.2   DeviceWrapper, device=/gpu:0
    WARNING:tensorflow:From /Users/shinkeika/learning/nlp_project/4-chatbot/tensorflow-seq2seq-chatbot/nmt/nmt/model_helper.py:508: MultiRNNCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This class is equivalent as tf.keras.layers.StackedRNNCells, and will be replaced by that in Tensorflow 2.0.
    WARNING:tensorflow:From /Users/shinkeika/learning/nlp_project/4-chatbot/tensorflow-seq2seq-chatbot/nmt/nmt/model.py:767: dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `keras.layers.RNN(cell)`, which is equivalent to this API
    WARNING:tensorflow:From /Users/shinkeika/anaconda3/lib/python3.7/site-packages/tensorflow/python/ops/rnn.py:626: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    WARNING:tensorflow:From /Users/shinkeika/anaconda3/lib/python3.7/site-packages/tensorflow/python/ops/rnn_cell_impl.py:1259: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
    WARNING:tensorflow:From /Users/shinkeika/learning/nlp_project/4-chatbot/tensorflow-seq2seq-chatbot/nmt/nmt/model.py:445: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
      cell 0  LSTM, forget_bias=1  DropoutWrapper, dropout=0.2   DeviceWrapper, device=/gpu:0
      cell 1  LSTM, forget_bias=1  DropoutWrapper, dropout=0.2   DeviceWrapper, device=/gpu:0
      learning_rate=1, warmup_steps=0, warmup_scheme=t2t
      decay_scheme=, start_decay_step=12000, decay_steps 0, decay_factor 1
    # Trainable variables
    Format: <name>, <shape>, <(soft) device placement>
      embeddings/encoder/embedding_encoder:0, (56491, 128), /device:CPU:0
      embeddings/decoder/embedding_decoder:0, (50041, 128), /device:CPU:0
      dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0, (256, 512), /device:GPU:0
      dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0, (512,), /device:GPU:0
      dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0, (256, 512), /device:GPU:0
      dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0, (512,), /device:GPU:0
      dynamic_seq2seq/decoder/memory_layer/kernel:0, (128, 128), 
      dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0, (384, 512), /device:GPU:0
      dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0, (512,), /device:GPU:0
      dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0, (256, 512), /device:GPU:0
      dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0, (512,), /device:GPU:0
      dynamic_seq2seq/decoder/attention/luong_attention/attention_g:0, (), /device:GPU:0
      dynamic_seq2seq/decoder/attention/attention_layer/kernel:0, (256, 128), /device:GPU:0
      dynamic_seq2seq/decoder/output_projection/kernel:0, (128, 50041), /device:GPU:0
    # Creating eval graph ...
    # Build a basic encoder
      num_layers = 2, num_residual_layers=0
      cell 0  LSTM, forget_bias=1  DeviceWrapper, device=/gpu:0
      cell 1  LSTM, forget_bias=1  DeviceWrapper, device=/gpu:0
      cell 0  LSTM, forget_bias=1  DeviceWrapper, device=/gpu:0
      cell 1  LSTM, forget_bias=1  DeviceWrapper, device=/gpu:0
    # Trainable variables
    Format: <name>, <shape>, <(soft) device placement>
      embeddings/encoder/embedding_encoder:0, (56491, 128), /device:CPU:0
      embeddings/decoder/embedding_decoder:0, (50041, 128), /device:CPU:0
      dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0, (256, 512), /device:GPU:0
      dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0, (512,), /device:GPU:0
      dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0, (256, 512), /device:GPU:0
      dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0, (512,), /device:GPU:0
      dynamic_seq2seq/decoder/memory_layer/kernel:0, (128, 128), 
      dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0, (384, 512), /device:GPU:0
      dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0, (512,), /device:GPU:0
      dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0, (256, 512), /device:GPU:0
      dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0, (512,), /device:GPU:0
      dynamic_seq2seq/decoder/attention/luong_attention/attention_g:0, (), /device:GPU:0
      dynamic_seq2seq/decoder/attention/attention_layer/kernel:0, (256, 128), /device:GPU:0
      dynamic_seq2seq/decoder/output_projection/kernel:0, (128, 50041), /device:GPU:0
    # Creating infer graph ...
    # Build a basic encoder
      num_layers = 2, num_residual_layers=0
      cell 0  LSTM, forget_bias=1  DeviceWrapper, device=/gpu:0
      cell 1  LSTM, forget_bias=1  DeviceWrapper, device=/gpu:0
      cell 0  LSTM, forget_bias=1  DeviceWrapper, device=/gpu:0
      cell 1  LSTM, forget_bias=1  DeviceWrapper, device=/gpu:0
      decoder: infer_mode=greedybeam_width=0, length_penalty=0.000000, coverage_penalty=0.000000
    # Trainable variables
    Format: <name>, <shape>, <(soft) device placement>
      embeddings/encoder/embedding_encoder:0, (56491, 128), /device:CPU:0
      embeddings/decoder/embedding_decoder:0, (50041, 128), /device:CPU:0
      dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0, (256, 512), /device:GPU:0
      dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0, (512,), /device:GPU:0
      dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0, (256, 512), /device:GPU:0
      dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0, (512,), /device:GPU:0
      dynamic_seq2seq/decoder/memory_layer/kernel:0, (128, 128), 
      dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0, (384, 512), /device:GPU:0
      dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0, (512,), /device:GPU:0
      dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0, (256, 512), /device:GPU:0
      dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0, (512,), /device:GPU:0
      dynamic_seq2seq/decoder/attention/luong_attention/attention_g:0, (), /device:GPU:0
      dynamic_seq2seq/decoder/attention/attention_layer/kernel:0, (256, 128), /device:GPU:0
      dynamic_seq2seq/decoder/output_projection/kernel:0, (128, 50041), 
    # log_file=/tmp/nmt_attention_model/log_1575259278
      created train model with fresh parameters, time 0.28s
      created infer model with fresh parameters, time 0.14s
      # 14721
        src: 去 你 个 的
        ref: 去 你 个 的
        nmt: 小姐姐 友爱 友爱 友爱 愛徐 愛徐 一箭双雕 一箭双雕
      created eval model with fresh parameters, time 0.14s


