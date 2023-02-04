## 环境
Anaconda3 4.3.1
tensorflow 1.5.0  
keras 2.2.0

## 整体说明
1.整个任务被拆解为2个任务:主题分类任务和情感分类任务
2.主题分类
主题分类时多输出(10个独立的输出)，主题分类任务采用了3个算法，rnn+attention(自己想的，没有相应论文)，另外两个对应的论文是:
《Convolutional Neural Networks for Sentence Classification》
《Densely Connected CNN with Multi-scale Feature Attention for Text Classification》
最终，样本的每个主题的概率是这3个模型输出的平均，结果概率文件放在data/test_subject_probability.result.bagging_topic
3.主题分类完成后，会根据主题分类模型对样本进行处理:如果一个样本有多个主题，针对每个主题，只截取这个样本中描述这个主题的文本。方法是，用主题分类模型对样本中的每句话进行主题预测，把连续描述这个主题的句子并起来作为该主题对应的文本。最终用于情感分类的输入文件是:
data/test_public_for_sentiment_value.exact.word
data/train_sentiment_value.exact.word
data/val_sentiment_value.exact.word
在data目录下提供了这3个文件，方便直接运行情感分类模型
4.情感分类
情感分类是个3分类任务，和通常的3分类任务不同的点在于，同样的文本在不同的主题下类别可能不同，所欲分类时，不仅要以文本作为特征，还得考虑主题，基于此，实现了3篇主流基于方面的情感分类的论文:
《Attention-based LSTM for Aspect-level Sentiment Classification》
《Interactive Attention Networks for Aspect-Level Sentiment Classification》
《Aspect Level Sentiment Classification with Attention-over-Attention Neural Networks》
代码分别在
at_lstm\rnn_attention_cv.py
sentiment_classifier\interactive_attention\interactive_attention_cv.py
sentiment_classifier\aoa\aoa_cv.py
针对情感分类这一特定任务，对词向量做了调整。原来训练得到的词向量是100维，给每个词向量增加了1维，用于标识这个词是否为情感词，1表示是情感词，0表示不是。情感词来源于训练数据中提供的情感分词。
同时，为了增加集成模型的数量，三个模型的rnn分别用了lstm和gru，于是就得到6个模型，最终的结果是基于这6个模型加之前效果较好的提交文件投票的方式得到最终结果(类比随机森林集成方式)，原来用于集成的文件都已提供，可以运行脚本run_pipeline_best.sh得到最优输出结果，集成方式在sentiment_classifier\bagging\rf_2.py实现。

## 包介绍
1. conf                    包含所有的配置文件，要在自己的环境里运行代码，需要修改data_path.py里embedding_base_dir、embedding_file和data_base_dir，它们分别指定保存预训练词向量的目录、预训练词向量的文件名和保存训练数据等的目录，以及model_path.py里的model_base_dir，设置保存训练好的模型文件的目录
2. data                    目前这个目录里保存了训练数据、测试数据、各种中间文件以及模型预测结果
3. eda                     里面的ipynb文件包含了对数据的简单分析
4. models                  里面包含了定义keras层和完整keras模型的模块
5. preprocess              里面包含了对原始数据进行预处理，使之适合模型的输入
6. sentiment_classifier    情感分类的代码
7. topic_classifier        主题分类的代码
8. utils                   常用工具模块
9. subject_sentiment_value_classifier  同时分类主题和情感的30分类启动代码
10.visualization           对attention进行可视化的代码
11.train_word_vector       训练词向量的代码

快速开始
---
1. 配置data_path.py中的embedding_base_dir、embedding_file和data_base_dir以及model_path.py里的model_base_dir，
2. cd auto-industry-topic-sentiment
3. sh run_pipeline_subject.sh 主题分类
4. sh run_pipeline_sentiment.sh 情感分类(可以直接运行，中间文件已提供)
5. 提交文件: ${data_base_dir}/test_public.result_(年月起时分秒).submit.csv 示例 test_public.result_20180920084855.submit.csv
6. 提交文件：对结果文件增加备注说明   示例：great.csv
7. sh run_pipeline_best.sh 原来用于集成的文件都已提供，可以运行脚本run_pipeline_best.sh得到最优输出结果，集成方式在sentiment_classifier\bagging\rf_2.py实现。
## 其它
1. 使用的词向量是根据从https://www.pcauto.com.cn/抓取的语料训练的，维度为100维，放在项目的data目录下，名为qiche_embedding
2. 最优成绩是由之前提交的两个成绩较好的结果与六个模型预测的结果集成得到的，那两个之前提交的结果放在data/bagging3下面
