[TOC]

#### 跟着demo解读BERT 源码

- **那我们就跑官方demo吧**

  `这个demo就是分类任务使用MRPC训练数据`

  - 目录结构

    <img src="https://shinkeika.github.io/images/bert/0/1.png" alt="image-20191202141121957" style="zoom:50%;" />

    - bert-master bert项目
    - [BERT_BASE_DIR 里放google训练好的word-embedding](https://github.com/shinkeika/nlp_project/tree/master/0-bert-source_code/GLUE/BERT_BASE_DIR)
    - Glue_data里面放的是训练数据

  

  - 训练数据的格式

    <img src="https://shinkeika.github.io/images/bert/0/2.png" alt="image-20191202142125336" style="zoom:50%;" />

    - Quanlity 表示两句话是否表达一个意思 1代表相同，0代表不相同
    - \#1ID 代表第一句话的ID
    - \#2ID 代表第二句话的ID
    - \#1 String 代表的是第一句话
    - \#2 String 代表的是第二句话

  - 训练参数

    ```python
    python run_classifier.py \
      --task_name=MRPC \    # 任务名
      --do_train=true \		 # 是否训练
      --do_eval=true \		# 训练完是否验证
      --data_dir=/Users/shinkeika/learning/nlp_project/0-bert-source_code/GLUE/glue_data/MRPC/ \  # 数据集
      --vocab_file=/Users/shinkeika/learning/nlp_project/0-bert-source_code/GLUE/BERT_BASE_DIR/uncased_L-12_H-768_A-12/vocab.txt \  # wordembedding的词表文件
      --bert_config_file=$BERT_BASE_DIR/bert_config.json \  # wordembedding的超参数
      --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \  # 预训练的checkpoint
      --max_seq_length=128 \  # 句子最大长度
      --train_batch_size=32 \  # batch_size 机器不好，选小一点的步长
      --learning_rate=2e-5 \  # 学习率
      --num_train_epochs=3.0 \  # 跑几个epoch
      --output_dir=/tmp/mrpc_output/  # 输出到哪个文件夹
    ```

  - 从项目入口分析源码

    - 一些检查性的操作,验证数据

    ```python
    def main(_):
      tf.logging.set_verbosity(tf.logging.INFO)
    
      processors = {
          "cola": ColaProcessor,
          "mnli": MnliProcessor,
          "mrpc": MrpcProcessor,
          "xnli": XnliProcessor,
      }
    
      tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                    FLAGS.init_checkpoint)
      # TODO
      # 都是一些输入参数的验证工作
      if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")
    
      bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    
      if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))
    
      tf.gfile.MakeDirs(FLAGS.output_dir)
    
      task_name = FLAGS.task_name.lower()
    
      if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    
      processor = processors[task_name]()
    
      label_list = processor.get_labels()
    
      tokenizer = tokenization.FullTokenizer(
          vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    
      tpu_cluster_resolver = None
      if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    
      is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
      run_config = tf.contrib.tpu.RunConfig(
          cluster=tpu_cluster_resolver,
          master=FLAGS.master,
          model_dir=FLAGS.output_dir,
          save_checkpoints_steps=FLAGS.save_checkpoints_steps,
          tpu_config=tf.contrib.tpu.TPUConfig(
              iterations_per_loop=FLAGS.iterations_per_loop,
              num_shards=FLAGS.num_tpu_cores,
              per_host_input_for_training=is_per_host))
    
      train_examples = None
      num_train_steps = None
      num_warmup_steps = None
      if FLAGS.do_train:
        # TODO
        # 从这里开始读取数据
        train_examples = processor.get_train_examples(FLAGS.data_dir)
    
        # TODO
        # 计算需要迭代多少次
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    
        # TODO
        # 在刚开始时候，学习率偏少
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    
      model_fn = model_fn_builder(
          bert_config=bert_config,
          num_labels=len(label_list),
          init_checkpoint=FLAGS.init_checkpoint,
          learning_rate=FLAGS.learning_rate,
          num_train_steps=num_train_steps,
          num_warmup_steps=num_warmup_steps,
          use_tpu=FLAGS.use_tpu,
          use_one_hot_embeddings=FLAGS.use_tpu)
    
      # If TPU is not available, this will fall back to normal Estimator on CPU
      # or GPU.
      estimator = tf.contrib.tpu.TPUEstimator(
          use_tpu=FLAGS.use_tpu,
          model_fn=model_fn,
          config=run_config,
          train_batch_size=FLAGS.train_batch_size,
          eval_batch_size=FLAGS.eval_batch_size,
          predict_batch_size=FLAGS.predict_batch_size)
    
      if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        # TODO
        # 跳到这个方法
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    
      if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
          # TPU requires a fixed batch size for all batches, therefore the number
          # of examples must be a multiple of the batch size, or else examples
          # will get dropped. So we pad with fake examples which are ignored
          # later on. These do NOT count towards the metric (all tf.metrics
          # support a per-instance weight, and these get a weight of 0.0).
          while len(eval_examples) % FLAGS.eval_batch_size != 0:
            eval_examples.append(PaddingInputExample())
    
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
    
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
    
        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
          assert len(eval_examples) % FLAGS.eval_batch_size == 0
          eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)
    
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
    
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
          tf.logging.info("***** Eval results *****")
          for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    
      if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
          # TPU requires a fixed batch size for all batches, therefore the number
          # of examples must be a multiple of the batch size, or else examples
          # will get dropped. So we pad with fake examples which are ignored
          # later on.
          while len(predict_examples) % FLAGS.predict_batch_size != 0:
            predict_examples.append(PaddingInputExample())
    
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)
    
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
    
        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)
    
        result = estimator.predict(input_fn=predict_input_fn)
    
        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
          num_written_lines = 0
          tf.logging.info("***** Predict results *****")
          for (i, prediction) in enumerate(result):
            probabilities = prediction["probabilities"]
            if i >= num_actual_predict_examples:
              break
            output_line = "\t".join(
                str(class_probability)
                for class_probability in probabilities) + "\n"
            writer.write(output_line)
            num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples
    ```

    - 把每个输入数据转换为TF-record的格式

      ```python
      def file_based_convert_examples_to_features(
          examples, label_list, max_seq_length, tokenizer, output_file):
        """Convert a set of `InputExample`s to a TFRecord file."""
      
        # TODO
        # TFrecord处理数据快
      
        writer = tf.python_io.TFRecordWriter(output_file)
      
        for (ex_index, example) in enumerate(examples):
          if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
      
          # TODO
          # 跳进
          feature = convert_single_example(ex_index, example, label_list,
                                           max_seq_length, tokenizer)
      
          def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f
      
          features = collections.OrderedDict()
          # TODO
          # 转换为int，方便TFrecord
          features["input_ids"] = create_int_feature(feature.input_ids)
          features["input_mask"] = create_int_feature(feature.input_mask)
          features["segment_ids"] = create_int_feature(feature.segment_ids)
          features["label_ids"] = create_int_feature([feature.label_id])
          features["is_real_example"] = create_int_feature(
              [int(feature.is_real_example)])
          # TODO
          # 转为tf train的格式
          tf_example = tf.train.Example(features=tf.train.Features(feature=features))
          # 写recorder
          writer.write(tf_example.SerializeToString())
        writer.close()
      ```

    - 核心构建数据集的代码,针对其子代码进行分析。

    - ```python
      def convert_single_example(ex_index, example, label_list, max_seq_length,
                                 tokenizer):
        """Converts a single `InputExample` into a single `InputFeatures`."""
      
        if isinstance(example, PaddingInputExample):
          return InputFeatures(
              input_ids=[0] * max_seq_length,
              input_mask=[0] * max_seq_length,
              segment_ids=[0] * max_seq_length,
              label_id=0,
              is_real_example=False)
        # TODO
        # 转换为数据标签，0和1 字典格式
        label_map = {}
        for (i, label) in enumerate(label_list):
          label_map[label] = i
        # TODO
        # 第一句话做了个wordpiece分词
        # 跳
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
      
        # 如果有第二句话
        if example.text_b:
          tokens_b = tokenizer.tokenize(example.text_b)
      
        # TODO
        # 如果有第二句话
        # cls [  第一句话  ] seg [    第二句话    ] seg
        # 如果有一句话
        # cls [ 第一句话  ] seg
        if tokens_b:
          # Modifies `tokens_a` and `tokens_b` in place so that the total
          # length is less than the specified length.
          # Account for [CLS], [SEP], [SEP] with "- 3"
          # TODO
          # 如果有b就保留三个特殊字符
          # 截断操作
          _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
          # Account for [CLS] and [SEP] with "- 2"
          if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]
      
        # TODO
        # 解释一下
        # 我们的输入是两句话，以下格式
        # tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        # typeid 为0 代表前一句话，为1 代表后一句话
        # type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
      
      
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
      
        # TODO
        # 第一句话加一个cls
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
      
            # 组合数据
          tokens.append(token)
            # 第一句话都是0
          segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
      
        if tokens_b:
            # 如果有b
            # 第二句话都是1
          for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
          tokens.append("[SEP]")
          segment_ids.append(1)
      
        # TODO
        # 把词转换成索引，通过查找vocab的词表
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
      
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # TODO
        # mask 为1 的代表为真的词，补齐的都是0
        input_mask = [1] * len(input_ids)
      
        # Zero-pad up to the sequence length.
        # TODO
        # 做一个截断补齐操作
        while len(input_ids) < max_seq_length:
          input_ids.append(0)
          input_mask.append(0)
          segment_ids.append(0)
      
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
      
        label_id = label_map[example.label]
        if ex_index < 5:
          tf.logging.info("*** Example ***")
          tf.logging.info("guid: %s" % (example.guid))
          tf.logging.info("tokens: %s" % " ".join(
              [tokenization.printable_text(x) for x in tokens]))
          tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
          tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
          tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
          tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
      
        # TODO
        # 跳
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id,
            is_real_example=True)
        return feature
      ```

    - 转换为wordpiece

      ```python
      def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
          # TODO
          # 词切片，把引文词切分成更基础的单元
          # 中文一般切分成字
          for sub_token in self.wordpiece_tokenizer.tokenize(token):
            split_tokens.append(sub_token)
      ```

    - 创建模型

      ```python
      # 创建模型
      def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                       labels, num_labels, use_one_hot_embeddings):
        """Creates a classification model."""
        # TODO
        # 跳 modeling.py的107行
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,  # （16，128） # 16 是 batch size
            input_mask=input_mask,  # （16，128）
            token_type_ids=segment_ids,  #（0，128）
            use_one_hot_embeddings=use_one_hot_embeddings)  # 没TPU就False
      
        # In the demo, we are doing a simple classification task on the entire
        # segment.
        #
        # If you want to use the token-level output, use model.get_sequence_output()
        # instead.
        # TODO
        # cls [第一句话] seg [第二句话] seg  所以只需要取第一个
        output_layer = model.get_pooled_output()
      
        hidden_size = output_layer.shape[-1].value
      
        # TODO
        # 二分类
        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
      
        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())
          # TODO
          # 常规操作
        with tf.variable_scope("loss"):
          if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
      
          logits = tf.matmul(output_layer, output_weights, transpose_b=True)
          logits = tf.nn.bias_add(logits, output_bias)
          probabilities = tf.nn.softmax(logits, axis=-1)
          log_probs = tf.nn.log_softmax(logits, axis=-1)
      
          one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
      
          per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
          loss = tf.reduce_mean(per_example_loss)
      
          return (loss, per_example_loss, logits, probabilities)
      ```

    - bert model核心

      ~~~python
      class BertModel(object):
        """BERT model ("Bidirectional Encoder Representations from Transformers").
      
        Example usage:
      
        ```python
        # Already been converted into WordPiece token ids
        input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
        input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
        token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])
      
        config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
          num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)
      
        model = modeling.BertModel(config=config, is_training=True,
          input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)
      
        label_embeddings = tf.get_variable(...)
        pooled_output = model.get_pooled_output()
        logits = tf.matmul(pooled_output, label_embeddings)
        ...
        ```
        """
      
        def __init__(self,
                     config,
                     is_training,
                     input_ids,
                     input_mask=None,
                     token_type_ids=None,
                     use_one_hot_embeddings=False,
                     scope=None):
          """Constructor for BertModel.
      
          Args:
            config: `BertConfig` instance.
            is_training: bool. true for training model, false for eval model. Controls
              whether dropout will be applied.
            input_ids: int32 Tensor of shape [batch_size, seq_length].
            input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
            token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
            use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
              embeddings or tf.embedding_lookup() for the word embeddings.
            scope: (optional) variable scope. Defaults to "bert".
      
          Raises:
            ValueError: The config is invalid or one of the input tensor shapes
              is invalid.
          """
          config = copy.deepcopy(config)
          if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
          # TODO
          # shape
          input_shape = get_shape_list(input_ids, expected_rank=2)
          batch_size = input_shape[0]
          seq_length = input_shape[1]
      
          # TODO
          # 没做mask的自动添加mask，加的1
          if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)
      
          # TODO
          # 没说有两句话，默认一句话加 typeid
          if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
      
          with tf.variable_scope(scope, default_name="bert"):
            with tf.variable_scope("embeddings"):
              # Perform embedding lookup on the word ids.
              # TODO
              # embedding层
              # 跳
              (self.embedding_output, self.embedding_table) = embedding_lookup(
                  input_ids=input_ids,  #
                  vocab_size=config.vocab_size,  # 词表维度
                  embedding_size=config.hidden_size,  # embedding维度 默认768
                  initializer_range=config.initializer_range,  # 初始化取值范围
                  word_embedding_name="word_embeddings",
                  use_one_hot_embeddings=use_one_hot_embeddings)  # 用不用one-hot TPU才用
      
              # TODO
              # 加上position encoding 都是和embedding同样的维度
              # 跳
              # Add positional embeddings and token type embeddings, then layer
              # normalize and perform dropout.
              self.embedding_output = embedding_postprocessor(
                  input_tensor=self.embedding_output,
                  use_token_type=True,
                  token_type_ids=token_type_ids,
                  token_type_vocab_size=config.type_vocab_size,
                  token_type_embedding_name="token_type_embeddings",
                  use_position_embeddings=True,
                  position_embedding_name="position_embeddings",
                  initializer_range=config.initializer_range,
                  max_position_embeddings=config.max_position_embeddings,
                  dropout_prob=config.hidden_dropout_prob)
      
            with tf.variable_scope("encoder"):
              # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
              # mask of shape [batch_size, seq_length, seq_length] which is used
              # for the attention scores.
              # TODO
              # 把2Dmask转换为3Dmask
              # 创建了mask矩阵
              # 16个batch [
              # 12,32,4,5,6,2,100,123....,
              # 1,32,44,55,1,5,101,145....
              # ]
              # 查看该句话的每个词需要和哪个词做计算
              # 拓展了一个维度
              # (16,128)(16,128,128) 最后一个128 表示某个词需要看哪些词
              attention_mask = create_attention_mask_from_input_mask(
                  input_ids, input_mask)
      
              # Run the stacked transformer.
              # `sequence_output` shape = [batch_size, seq_length, hidden_size].
              # TODO
              # 跳
              self.all_encoder_layers = transformer_model(
                  input_tensor=self.embedding_output,  # 之前三种融合
                  attention_mask=attention_mask,  # 有些映射到0，有些映射到1
                  hidden_size=config.hidden_size,  # 最终希望得到的特征结果
                  num_hidden_layers=config.num_hidden_layers,
                  num_attention_heads=config.num_attention_heads,
                  intermediate_size=config.intermediate_size,
                  intermediate_act_fn=get_activation(config.hidden_act),
                  hidden_dropout_prob=config.hidden_dropout_prob,
                  attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                  initializer_range=config.initializer_range,
                  do_return_all_layers=True)
            # TODO
            # 最后一个是结果
            self.sequence_output = self.all_encoder_layers[-1]
            # The "pooler" converts the encoded sequence tensor of shape
            # [batch_size, seq_length, hidden_size] to a tensor of shape
            # [batch_size, hidden_size]. This is necessary for segment-level
            # (or segment-pair-level) classification tasks where we need a fixed
            # dimensional representation of the segment.
            with tf.variable_scope("pooler"):
              # We "pool" the model by simply taking the hidden state corresponding
              # to the first token. We assume that this has been pre-trained
              # TODO
              # 看你想返回什么，cls
              first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
              self.pooled_output = tf.layers.dense(
                  first_token_tensor,
                  config.hidden_size,
                  activation=tf.tanh,
                  kernel_initializer=create_initializer(config.initializer_range))
      
        def get_pooled_output(self):
          return self.pooled_output
      
        def get_sequence_output(self):
          """Gets final hidden layer of encoder.
      
          Returns:
            float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
            to the final hidden of the transformer encoder.
          """
          return self.sequence_output
      
        def get_all_encoder_layers(self):
          return self.all_encoder_layers
      
        def get_embedding_output(self):
          """Gets output of the embedding lookup (i.e., input to the transformer).
      
          Returns:
            float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
            to the output of the embedding layer, after summing the word
            embeddings with the positional embeddings and the token type embeddings,
            then performing layer normalization. This is the input to the transformer.
          """
          return self.embedding_output
      
        def get_embedding_table(self):
          return self.embedding_table
      ~~~

    - embedding_lookup

      ```python
      def embedding_lookup(input_ids,
                           vocab_size,
                           embedding_size=128,
                           initializer_range=0.02,
                           word_embedding_name="word_embeddings",
                           use_one_hot_embeddings=False):
        """Looks up words embeddings for id tensor.
        # TODO
        # input [batch_size, seq_length]
        # 多了个编码维度
        # 返回值 [batch_size, seq_length, embedding_size]
      
        Args:
          input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
            ids.
          vocab_size: int. Size of the embedding vocabulary.
          embedding_size: int. Width of the word embeddings.
          initializer_range: float. Embedding initialization range.
          word_embedding_name: string. Name of the embedding table.
          use_one_hot_embeddings: bool. If True, use one-hot method for word
            embeddings. If False, use `tf.gather()`.
      
        Returns:
          float Tensor of shape [batch_size, seq_length, embedding_size].
        """
        # This function assumes that the input is of shape [batch_size, seq_length,
        # num_inputs].
        #
        # If the input is a 2D tensor of shape [batch_size, seq_length], we
        # reshape to [batch_size, seq_length, 1].
      
        if input_ids.shape.ndims == 2:
          input_ids = tf.expand_dims(input_ids, axis=[-1])
      
        embedding_table = tf.get_variable(
            name=word_embedding_name,
            shape=[vocab_size, embedding_size],
            initializer=create_initializer(initializer_range))
        # 一次查找的 16 128
        flat_input_ids = tf.reshape(input_ids, [-1])
        if use_one_hot_embeddings:
          one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
          output = tf.matmul(one_hot_input_ids, embedding_table)
        else:
          # 查当前 [batchsize和maxlength] 的维度
          output = tf.gather(embedding_table, flat_input_ids)
      
        input_shape = get_shape_list(input_ids)
        # TODO
        # 结果的维度 [batchsize，词数，embedding维度]
        output = tf.reshape(output,
                            input_shape[0:-1] + [input_shape[-1] * embedding_size])
        return (output, embedding_table)
      ```

    - Position encoding

      ```python
      def embedding_postprocessor(input_tensor,
                                  use_token_type=False,
                                  token_type_ids=None,
                                  token_type_vocab_size=16,
                                  token_type_embedding_name="token_type_embeddings",
                                  use_position_embeddings=True,
                                  position_embedding_name="position_embeddings",
                                  initializer_range=0.02,
                                  max_position_embeddings=512,
                                  dropout_prob=0.1):
      
          # TODO
          # 输入参数，
          # 上步骤结果 embedding [batchsize,maxlength,embedding维度]
          # 不要不要typeid 表示第一句第二句
          # 位置信息
          # 位置信息长度
          # dropout
      
          # 返回和 输入一样的维度
        """Performs various post-processing on a word embedding tensor.
      
        Args:
          input_tensor: float Tensor of shape [batch_size, seq_length,
            embedding_size].
          use_token_type: bool. Whether to add embeddings for `token_type_ids`.
          token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
            Must be specified if `use_token_type` is True.
          token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
          token_type_embedding_name: string. The name of the embedding table variable
            for token type ids.
          use_position_embeddings: bool. Whether to add position embeddings for the
            position of each token in the sequence.
          position_embedding_name: string. The name of the embedding table variable
            for positional embeddings.
          initializer_range: float. Range of the weight initialization.
          max_position_embeddings: int. Maximum sequence length that might ever be
            used with this model. This can be longer than the sequence length of
            input_tensor, but cannot be shorter.
          dropout_prob: float. Dropout probability applied to the final output tensor.
      
        Returns:
          float tensor with same shape as `input_tensor`.
      
        Raises:
          ValueError: One of the tensor shapes or input values is invalid.
        """
        input_shape = get_shape_list(input_tensor, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]
        # TODO
        # 维度不变
        output = input_tensor
      
        if use_token_type:
          if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if"
                             "`use_token_type` is True.")
          # TODO
          # 加不加type值
          # 只有两种可能性，是第一句还是第二句
          # [2, 768]
          token_type_table = tf.get_variable(
              name=token_type_embedding_name,
              shape=[token_type_vocab_size, width],
              initializer=create_initializer(initializer_range))
          # This vocab will be small so we always do one-hot here, since it is always
          # faster for a small vocabulary.
          # TODO
          # 一共有多少个词 16  * 128
          flat_token_type_ids = tf.reshape(token_type_ids, [-1])
          # TODO
          # one_hot 表明是第几句还是第二句
          one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
          token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
          token_type_embeddings = tf.reshape(token_type_embeddings,
                                             [batch_size, seq_length, width])
          # TODO
          # 把typeid信息融入到原始编码中
          output += token_type_embeddings
        # TODO
        # 位置编码信息
        if use_position_embeddings:
          assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
          with tf.control_dependencies([assert_op]):
            # TODO
            # 假设512个位置，每个位置都是768维度向量
            full_position_embeddings = tf.get_variable(
                name=position_embedding_name,
                shape=[max_position_embeddings, width],
                initializer=create_initializer(initializer_range))
            # Since the position embedding table is a learned variable, we create it
            # using a (long) sequence length `max_position_embeddings`. The actual
            # sequence length might be shorter than this, for faster training of
            # tasks that do not have long sequences.
            #
            # So `full_position_embeddings` is effectively an embedding table
            # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
            # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
            # perform a slice.
            # TODO
            # 根据序列长度 截取  （128）
            position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                           [seq_length, -1])
            num_dims = len(output.shape.as_list())
      
            # Only the last two dimensions are relevant (`seq_length` and `width`), so
            # we broadcast among the first dimensions, which is typically just
            # the batch size.
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
              position_broadcast_shape.append(1)
      
            position_broadcast_shape.extend([seq_length, width])
            # TODO
            # 位置在什么地方，1-128  1-128
            # 现在得到的128-768，把batch中的每一个加在一起
            # 位置编码结果和实际传进来的词无关
            position_embeddings = tf.reshape(position_embeddings,
                                             position_broadcast_shape)
            output += position_embeddings
        # TODO
        # layer normalization 和 dropout
        output = layer_norm_and_dropout(output, dropout_prob)
        return output
      ```

    - mask机制

    - ```python
      def create_attention_mask_from_input_mask(from_tensor, to_mask):
        """Create 3D attention mask from a 2D tensor mask.
      
        Args:
          from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
          to_mask: int32 Tensor of shape [batch_size, to_seq_length].
      
        Returns:
          float Tensor of shape [batch_size, from_seq_length, to_seq_length].
        """
        from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
      
        to_shape = get_shape_list(to_mask, expected_rank=2)
        to_seq_length = to_shape[1]
      
        to_mask = tf.cast(
            tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)
      
        # We don't assume that `from_tensor` is a mask (although it could be). We
        # don't actually care if we attend *from* padding tokens (only *to* padding)
        # tokens so we create a tensor of all ones.
        #
        # `broadcast_ones` = [batch_size, from_seq_length, 1]
        broadcast_ones = tf.ones(
            shape=[batch_size, from_seq_length, 1], dtype=tf.float32)
      
        # Here we broadcast along two dimensions to create the mask.
        mask = broadcast_ones * to_mask
      
        return mask
      ```

    - transformer

      ```python
      def transformer_model(input_tensor,
                            attention_mask=None,
                            hidden_size=768,
                            num_hidden_layers=12,
                            num_attention_heads=12,
                            intermediate_size=3072,
                            intermediate_act_fn=gelu,
                            hidden_dropout_prob=0.1,
                            attention_probs_dropout_prob=0.1,
                            initializer_range=0.02,
                            do_return_all_layers=False):
        """Multi-headed, multi-layer Transformer from "Attention is All You Need".
      
        This is almost an exact implementation of the original Transformer encoder.
      
        See the original paper:
        https://arxiv.org/abs/1706.03762
      
        Also see:
        https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
      
        Args:
          input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
          attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
            seq_length], with 1 for positions that can be attended to and 0 in
            positions that should not be.
          hidden_size: int. Hidden size of the Transformer.
          num_hidden_layers: int. Number of layers (blocks) in the Transformer.
          num_attention_heads: int. Number of attention heads in the Transformer.
          intermediate_size: int. The size of the "intermediate" (a.k.a., feed
            forward) layer.
          intermediate_act_fn: function. The non-linear activation function to apply
            to the output of the intermediate/feed-forward layer.
          hidden_dropout_prob: float. Dropout probability for the hidden layers.
          attention_probs_dropout_prob: float. Dropout probability of the attention
            probabilities.
          initializer_range: float. Range of the initializer (stddev of truncated
            normal).
          do_return_all_layers: Whether to also return all layers or just the final
            layer.
      
        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size], the final
          hidden layer of the Transformer.
      
        Raises:
          ValueError: A Tensor shape or parameter is invalid.
        """
        # TODO
        # hidden_size 希望得到的结果
        # 768 / 12 = 64 需要整除，如果不能整除就报错
        if hidden_size % num_attention_heads != 0:
          raise ValueError(
              "The hidden size (%d) is not a multiple of the number of attention "
              "heads (%d)" % (hidden_size, num_attention_heads))
      
        attention_head_size = int(hidden_size / num_attention_heads)
        input_shape = get_shape_list(input_tensor, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        input_width = input_shape[2]
      
        # The Transformer performs sum residuals on all layers so the input needs
        # to be the same as the hidden size.
        # TODO
        # 如果768特征和输入的是否一致，如果不一致报错
        # 残差连接，是加法操作
        if input_width != hidden_size:
          raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                           (input_width, hidden_size))
      
        # We keep the representation as a 2D tensor to avoid re-shaping it back and
        # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
        # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
        # help the optimizer.
        # TODO
        # 为了加速
        prev_output = reshape_to_matrix(input_tensor)
      
        all_layer_outputs = []
        # TODO
        # 遍历多少attention层
      
        for layer_idx in range(num_hidden_layers):
          with tf.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output
      
            with tf.variable_scope("attention"):
              attention_heads = []
              # TODO
              # self attention
              # 表示第一句只和第一句的attention
              # 跳
              with tf.variable_scope("self"):
                attention_head = attention_layer(
                    from_tensor=layer_input,
                    to_tensor=layer_input,  # from_tensor 和 to_tensor 相同代表是self attention
                    attention_mask=attention_mask,  # 1 和 0
                    num_attention_heads=num_attention_heads,  # 多少头
                    size_per_head=attention_head_size,  # 每个头多少特征
                    attention_probs_dropout_prob=attention_probs_dropout_prob,
                    initializer_range=initializer_range,
                    do_return_2d_tensor=True,  # 是否返回2Dtensor
                    batch_size=batch_size,
                    from_seq_length=seq_length,
                    to_seq_length=seq_length)
                attention_heads.append(attention_head)
      
              attention_output = None
              if len(attention_heads) == 1:
                attention_output = attention_heads[0]
              else:
                # In the case where we have other sequences, we just concatenate
                # them to the self-attention head before the projection.
                attention_output = tf.concat(attention_heads, axis=-1)
      
              # Run a linear projection of `hidden_size` then add a residual
              # with `layer_input`.
              with tf.variable_scope("output"):
                  # TODO
                  # 全连接层
                attention_output = tf.layers.dense(
                    attention_output,
                    hidden_size,
                    kernel_initializer=create_initializer(initializer_range))
                attention_output = dropout(attention_output, hidden_dropout_prob)
                  # TODO
                  # 残差连接
                attention_output = layer_norm(attention_output + layer_input)
      
            # The activation is only applied to the "intermediate" hidden layer.
            # TODO
            # 全连接了之后3072维度，特征多了，下一层，-> 768
            with tf.variable_scope("intermediate"):
              intermediate_output = tf.layers.dense(
                  attention_output,
                  intermediate_size,
                  activation=intermediate_act_fn,
                  kernel_initializer=create_initializer(initializer_range))
      
            # Down-project back to `hidden_size` then add the residual.
            with tf.variable_scope("output"):
              layer_output = tf.layers.dense(
                  intermediate_output,
                  hidden_size,
                  kernel_initializer=create_initializer(initializer_range))
              layer_output = dropout(layer_output, hidden_dropout_prob)
              layer_output = layer_norm(layer_output + attention_output)
              prev_output = layer_output
              all_layer_outputs.append(layer_output)
        # TODO
        # 是否返回所有层
        if do_return_all_layers:
          final_outputs = []
          for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
          return final_outputs
        else:
          final_output = reshape_from_matrix(prev_output, input_shape)
          return final_output
      ```

    - Attention机制

      ```python
      def attention_layer(from_tensor,
                          to_tensor,
                          attention_mask=None,
                          num_attention_heads=1,
                          size_per_head=512,
                          query_act=None,
                          key_act=None,
                          value_act=None,
                          attention_probs_dropout_prob=0.0,
                          initializer_range=0.02,
                          do_return_2d_tensor=False,
                          batch_size=None,
                          from_seq_length=None,
                          to_seq_length=None):
        # TODO
        # 注释
        """Performs multi-headed attention from `from_tensor` to `to_tensor`.
      
        This is an implementation of multi-headed attention based on "Attention
        is all you Need". If `from_tensor` and `to_tensor` are the same, then
        this is self-attention. Each timestep in `from_tensor` attends to the
        corresponding sequence in `to_tensor`, and returns a fixed-with vector.
      
        This function first projects `from_tensor` into a "query" tensor and
        `to_tensor` into "key" and "value" tensors. These are (effectively) a list
        of tensors of length `num_attention_heads`, where each tensor is of shape
        [batch_size, seq_length, size_per_head].
      
        Then, the query and key tensors are dot-producted and scaled. These are
        softmaxed to obtain attention probabilities. The value tensors are then
        interpolated by these probabilities, then concatenated back to a single
        tensor and returned.
      
        In practice, the multi-headed attention are done with transposes and
        reshapes rather than actual separate tensors.
      
        Args:
          from_tensor: float Tensor of shape [batch_size, from_seq_length,
            from_width].
          to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
          attention_mask: (optional) int32 Tensor of shape [batch_size,
            from_seq_length, to_seq_length]. The values should be 1 or 0. The
            attention scores will effectively be set to -infinity for any positions in
            the mask that are 0, and will be unchanged for positions that are 1.
          num_attention_heads: int. Number of attention heads.
          size_per_head: int. Size of each attention head.
          query_act: (optional) Activation function for the query transform.
          key_act: (optional) Activation function for the key transform.
          value_act: (optional) Activation function for the value transform.
          attention_probs_dropout_prob: (optional) float. Dropout probability of the
            attention probabilities.
          initializer_range: float. Range of the weight initializer.
          do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
            * from_seq_length, num_attention_heads * size_per_head]. If False, the
            output will be of shape [batch_size, from_seq_length, num_attention_heads
            * size_per_head].
          batch_size: (Optional) int. If the input is 2D, this might be the batch size
            of the 3D version of the `from_tensor` and `to_tensor`.
          from_seq_length: (Optional) If the input is 2D, this might be the seq length
            of the 3D version of the `from_tensor`.
          to_seq_length: (Optional) If the input is 2D, this might be the seq length
            of the 3D version of the `to_tensor`.
      
        Returns:
          float Tensor of shape [batch_size, from_seq_length,
            num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
            true, this will be of shape [batch_size * from_seq_length,
            num_attention_heads * size_per_head]).
      
        Raises:
          ValueError: Any of the arguments or tensor shapes are invalid.
        """
      
        def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                                 seq_length, width):
          output_tensor = tf.reshape(
              input_tensor, [batch_size, seq_length, num_attention_heads, width])
      
          output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
          return output_tensor
      
        # [2048,768]
      
        from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
        to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])
      
        if len(from_shape) != len(to_shape):
          raise ValueError(
              "The rank of `from_tensor` must match the rank of `to_tensor`.")
      
        if len(from_shape) == 3:
          batch_size = from_shape[0]
          from_seq_length = from_shape[1]
          to_seq_length = to_shape[1]
        elif len(from_shape) == 2:
          if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")
      
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`
      
        from_tensor_2d = reshape_to_matrix(from_tensor)
        to_tensor_2d = reshape_to_matrix(to_tensor)
      
        # `query_layer` = [B*F, N*H]
        # TODO
        # 查询矩阵，由from tensor 产生
        # [16*128, 64*12]
      
        query_layer = tf.layers.dense(
            from_tensor_2d,
            num_attention_heads * size_per_head,
            activation=query_act,
            name="query",
            kernel_initializer=create_initializer(initializer_range))
      
        # `key_layer` = [B*T, N*H]
        # TODO
        # 由我产生query totensor产生key和value
        # [16*128, 64*12]
        key_layer = tf.layers.dense(
            to_tensor_2d,
            num_attention_heads * size_per_head,
            activation=key_act,
            name="key",
            kernel_initializer=create_initializer(initializer_range))
      
        # `value_layer` = [B*T, N*H]
        # [16*128, 64*12]
        value_layer = tf.layers.dense(
            to_tensor_2d,
            num_attention_heads * size_per_head,
            activation=value_act,
            name="value",
            kernel_initializer=create_initializer(initializer_range))
      
        # `query_layer` = [B, N, F, H]
        # TODO
        # 做了个transpose 加速计算内积
        query_layer = transpose_for_scores(query_layer, batch_size,
                                           num_attention_heads, from_seq_length,
                                           size_per_head)
      
        # `key_layer` = [B, N, T, H]
        key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                         to_seq_length, size_per_head)
      
        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # `attention_scores` = [B, N, F, T]
        # TODO
        # 内积的结果值
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
      
        # TODO
        # 消除维度对结果的影响
        attention_scores = tf.multiply(attention_scores,
                                       1.0 / math.sqrt(float(size_per_head)))
      
        # TODO
        # 做softmax之前
        if attention_mask is not None:
          # `attention_mask` = [B, 1, F, T]
          # TODO
          # 每个词对128进行计算
          attention_mask = tf.expand_dims(attention_mask, axis=[1])
      
          # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
          # masked positions, this operation will create a tensor which is 0.0 for
          # positions we want to attend and -10000.0 for masked positions.
          # TODO
          # mask只有1和0，当mask为1-1 = 0 当mask为0 0-1 = -1 * 10000 = -10000
          # 1 softmax0 = 1 ， 0 softmax 之后接近0
          adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
      
          # Since we are adding it to the raw scores before the softmax, this is
          # effectively the same as removing these entirely.
          # TODO
          # adder 和得分值加在一起
          attention_scores += adder
      
        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        # TODO
        # softmax，所有得分值。得到概率。
        attention_probs = tf.nn.softmax(attention_scores)
      
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = dropout(attention_probs, attention_probs_dropout_prob)
      
        # `value_layer` = [B, T, N, H]
        value_layer = tf.reshape(
            value_layer,
            [batch_size, to_seq_length, num_attention_heads, size_per_head])
        # TODO
        # reshape 成一个和权重相乘的矩阵，和权重相同
        # 这里transpose 都是为了加速
        # `value_layer` = [B, N, T, H]
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
      
        # `context_layer` = [B, N, F, H]
        context_layer = tf.matmul(attention_probs, value_layer)
      
        # `context_layer` = [B, F, N, H]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
      
        if do_return_2d_tensor:
          # `context_layer` = [B*F, N*H]
          context_layer = tf.reshape(
              context_layer,
              [batch_size * from_seq_length, num_attention_heads * size_per_head])
        else:
          # `context_layer` = [B, F, N*H]
          context_layer = tf.reshape(
              context_layer,
              [batch_size, from_seq_length, num_attention_heads * size_per_head])
      
        return context_layer
      ```