### 使用tensorflow训练自己的word2vec

- 训练语料，因为是无监督训练，所以只要是通顺的句子都可以作为语料。

- 我们使用 skipgram 使用中间的词去预测上下文。

- 目录结构

  <img src="https://shinkeika.github.io/images/bert/3/0.png" alt="image-20191130102715254" style="zoom:50%;" />

  - text8 是训练数据
  - Word2vec.py 是训练文件

- 代码分析

  - 下载并返回文件名<img src="https://shinkeika.github.io/images/bert/3/1.png" alt="image-20191130103614453" style="zoom:50%;" />

  - 读取语料

    <img src="https://shinkeika.github.io/images/bert/3/2.png" alt="image-20191130103727830" style="zoom:50%;" />

  - 构建由词到索引和索引到词的dict

    <img src="https://shinkeika.github.io/images/bert/3/3.png" alt="image-20191130111855244" style="zoom:50%;" />

  - 根据滑动窗口，制作数据

    <img src="https://shinkeika.github.io/images/bert/3/4.png" alt="image-20191130115504850" style="zoom:50%;" />

  - 使用nceloss 负采样模型训练<img src="https://shinkeika.github.io/images/bert/3/5.png" alt="image-20191130120158791" style="zoom:50%;" />

  - 训练<img src="https://shinkeika.github.io/images/bert/3/9.png" alt="image-20191130121117372" style="zoom:50%;" />

- 可视化展示

  在项目根目录执行

  Tensorboard --logdir log

  使用tensorboard打开训练的记录

  <img src="https://shinkeika.github.io/images/bert/3/6.png" alt="image-20191130121737502" style="zoom:50%;" />

  然后在浏览器输入 http://localhost:6006/#scalars

  可以看到训练的损失

  <img src="https://shinkeika.github.io/images/bert/3/7.png" alt="image-20191130121908044" style="zoom:50%;" />

  在projector中看到可视化的词向量

<img src="https://shinkeika.github.io/images/bert/3/8.png" alt="image-20191130122148495" style="zoom:50%;" />