import os

from nlp_tasks.absa.conf import task_conf

# embedding_base_dir = r'D:\program\ml\machine-learning-databases\kaggle\Toxic Comment Classification Challenge\\'
# embedding_base_dir = r'D:\program\Chinese-Word-Vectors\\'
# embedding_base_dir = os.getcwd() + '/data/'

data_base_dir = r'D:\Users\hello_world\PycharmProjects\absa\data\\' + task_conf.current_dataset + '/'
# data_base_dir = os.getcwd() + '/data/' + task_conf.current_dataset + '/'

# bert_base_dir = data_base_dir + 'chinese_L-12_H-768_A-12/'
# bert_base_dir = os.getcwd() + '/data/'  + 'pretrain_data/uncased_L-12_H-768_A-12/'
bert_base_dir = r'D:\Users\hello_world\PycharmProjects\absa\data\uncased_L-12_H-768_A-12/'
bert_config_path = bert_base_dir + 'bert_config.json'
bert_checkpoint_path = bert_base_dir + 'bert_model.ckpt'
bert_dict_path = bert_base_dir + 'vocab.txt'

tf_bert_data = data_base_dir + '/tf_bert_data/'

train_original_file_path = data_base_dir + 'train.csv'
train_file_path = data_base_dir + 'train_train.csv'
val_file_path = data_base_dir + 'train_val.csv'
test_public_file_path = data_base_dir + 'test_public.csv'
test_public_gold_file_path = data_base_dir + 'test_public_gold.csv'
submit_example_file_path = data_base_dir + 'submit_example_2.csv'

test_public_with_label_file_path = data_base_dir + 'test_public_with_label.csv'

train_subject_file_path = data_base_dir + 'train_subject.csv'
train_sentiment_value_file_path = data_base_dir + 'train_sentiment_value.csv'
train_subject_sentiment_value_file_path = data_base_dir + 'train_subject_sentiment_value.csv'

train_subject_word_file_path = data_base_dir + 'train_subject.word'
train_sentiment_value_word_file_path = data_base_dir + 'train_sentiment_value.word'
train_sentiment_value_exact_file_path = data_base_dir + 'train_sentiment_value.exact'
train_sentiment_value_exact_word_file_path = data_base_dir + 'train_sentiment_value.exact.word'
train_subject_sentiment_value_word_file_path = data_base_dir + 'train_subject_sentiment_value.word'

train_subject_char_file_path = data_base_dir + 'train_subject.char'
train_sentiment_value_char_file_path = data_base_dir + 'train_sentiment_value.char'
train_subject_sentiment_value_char_file_path = data_base_dir + 'train_subject_sentiment_value.char'

train_subject_bigram_file_path = data_base_dir + 'train_subject.bigram'
train_sentiment_value_bigram_file_path = data_base_dir + 'train_sentiment_value.bigram'
train_subject_sentiment_value_bigram_file_path = data_base_dir + 'train_subject_sentiment_value.bigram'

train_subject_result_file_path = data_base_dir + 'train_subject.result'
train_sentiment_value_result_file_path = data_base_dir + 'train_sentiment_value.result'

val_subject_file_path = data_base_dir + 'val_subject.csv'
val_sentiment_value_file_path = data_base_dir + 'val_sentiment_value.csv'
val_subject_sentiment_value_file_path = data_base_dir + 'val_subject_sentiment_value.csv'

val_subject_word_file_path = data_base_dir + 'val_subject.word'
val_sentiment_value_word_file_path = data_base_dir + 'val_sentiment_value.word'
val_sentiment_value_exact_file_path = data_base_dir + 'val_sentiment_value.exact'
val_sentiment_value_exact_word_file_path = data_base_dir + 'val_sentiment_value.exact.word'
val_subject_sentiment_value_word_file_path = data_base_dir + 'val_subject_sentiment_value.word'

val_subject_char_file_path = data_base_dir + 'val_subject.char'
val_sentiment_value_char_file_path = data_base_dir + 'val_sentiment_value.char'
val_subject_sentiment_value_char_file_path = data_base_dir + 'val_subject_sentiment_value.char'

val_subject_bigram_file_path = data_base_dir + 'val_subject.bigram'
val_sentiment_value_bigram_file_path = data_base_dir + 'val_sentiment_value.bigram'
val_subject_sentiment_value_bigram_file_path = data_base_dir + 'val_subject_sentiment_value.bigram'

val_subject_result_file_path = data_base_dir + 'val_subject.result'
val_sentiment_value_result_file_path = data_base_dir + 'val_sentiment_value.result'

val_subject_probability_result_file_path = data_base_dir + 'val_subject_probability.result'
val_sentiment_value_probability_result_file_path = data_base_dir + 'val_sentiment_value_probability.result'

val_sentiment_value_result_file_path = data_base_dir + 'val_sentiment_value.result'

test_subject_file_path = data_base_dir + 'test_subject.csv'
test_sentiment_value_file_path = data_base_dir + 'test_sentiment_value.csv'

test_subject_word_file_path = data_base_dir + 'test_subject.word'
test_subject_char_file_path = data_base_dir + 'test_subject.char'
test_subject_bigram_file_path = data_base_dir + 'test_subject.bigram'

test_public_sentiment_word_file_path = data_base_dir + 'test_public_sentiment.word'

test_public_for_sentiment_value_file_path = data_base_dir \
                                                 + 'test_public_for_sentiment_value'
test_public_for_sentiment_value_word_file_path = data_base_dir \
                                                 + 'test_public_for_sentiment_value.word'
test_public_for_sentiment_value_char_file_path = data_base_dir \
                                                 + 'test_public_for_sentiment_value.char'

test_public_for_sentiment_value_exact_file_path = data_base_dir + 'test_public_for_sentiment_value.exact'
test_public_for_sentiment_value_exact_word_file_path = data_base_dir + 'test_public_for_sentiment_value.exact.word'
test_public_xingneng_caozong_file_path = data_base_dir + 'test_public_xingneng_caozong'


test_public_sentiment_value_result_file_path = data_base_dir + 'test_public_sentiment_value.result'
test_subject_result_file_path = data_base_dir + 'test_subject.result'
test_public_subject_sentiment_value_result_file_path = data_base_dir + 'test_public_subject_sentiment_value.result'
test_public_result_file_path = data_base_dir + 'test_public.result'
test_public_subject_sentiment_value_probability_result_file_path = data_base_dir + 'test_public_subject_sentiment_value_probability.result'
test_subject_probability_result_file_path = data_base_dir + 'test_subject_probability.result'
test_public_sentiment_value_probability_result_file_path = data_base_dir + 'test_public_sentiment_value_probability.result'

all_word_file_path = 'words'
word_index_subject_file_path = data_base_dir + 'word_index_subject'
word_index_sentiment_file_path = 'word_index_sentiment'
char_index_sentiment_file_path = 'char_index_sentiment'
for_sentiment_visualization = 'for_sentiment_visualization'
my_dict = data_base_dir + 'my_dict'
stopwords_file_path = data_base_dir + 'stopwords.txt'

#外部情感数据
external_sentiment_data_qiche = data_base_dir + 'external_sentiment_data_qiche'

external_data_topic = data_base_dir + 'external_data_topic'

external_data_suffix = '.external'

sentiment_dict_file_path = data_base_dir + '大连理工大学情感词汇本体库.csv'

sentiment_word_file_path = data_base_dir + 'sentiment_word'