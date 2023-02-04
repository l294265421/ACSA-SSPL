import os
import warnings
import re

import numpy as np
from keras.preprocessing import text, sequence

from nlp_tasks.absa.conf import data_path, thresholds, model_path
from nlp_tasks.absa.utils import data_utils
from nlp_tasks.absa.utils import embedding_utils, result_utils, evaluate_utils, cv_utils
from nlp_tasks.absa.utils import file_utils, tokenizer_utils
from nlp_tasks.absa.models import densely_cnn_multi_label_model, keras_models, cnn_for_classification_multi_label_model
from nlp_tasks.absa.preprocess import prepare_data_for_nn_subject

np.random.seed(42)
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '8'

train_file_path = data_path.train_file_path
val_file_path = data_path.val_file_path
test_file_path = data_path.test_public_gold_file_path

train_data = file_utils.read_all_lines(train_file_path)
head = [train_data.pop(0)]
val_data = file_utils.read_all_lines(val_file_path)[1:]
test_data = file_utils.read_all_lines(test_file_path)[1:]

max_features = 30000
embed_size = 300
maxlen = 150

tokenizer = tokenizer_utils.get_tokenizer()
word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index) + 1)
embedding_matrix = embedding_utils.generate_embedding_matrix(word_index, nb_words, data_path.embedding_file,
                                                             embed_size)

model_names = [
    'rnn_attention_multi_label_cv_topic',
    # 'rnn_attention_multi_label_cv_topic.merge',
    # 'rnn_attention_multi_label_cv_pl_topic',
    # 'cnn_for_classification_multi_label_cv_topic',
    # 'densely_cnn_multi_label_cv_topic'
]
model_name_model = {
                    'rnn_attention_multi_label_cv_topic': keras_models.rnn_attention_multi_label,
                    # 'rnn_attention_multi_label_cv_topic.merge': keras_models.rnn_attention_multi_label,
                    # 'rnn_attention_multi_label_cv_pl_topic': keras_models.rnn_attention_multi_label,
                    # 'cnn_for_classification_multi_label_cv_topic': cnn_for_classification_multi_label_model.cnn_for_classification_multi_label,
                    # 'densely_cnn_multi_label_cv_topic': densely_cnn_multi_label_model.densely_cnn_multi_label
}
models = []
model_num = 1
for model_name in model_names:
    one_models = []
    for i in range(model_num):
        model_filepath = model_path.model_file_dir + model_name + '_' + str(i) + '.hdf5'
        model = model_name_model[model_name](maxlen, nb_words, embed_size, embedding_matrix, 10)
        model.load_weights(model_filepath)
        one_models.append(model)
    models.append(one_models)

# best_thresholds = [0.78, 0.28, 0.5, 0.52,  0.2, 0.66, 0.64, 0.48, 0.58, 0.42]
# thresholds.topic_positive_threshold = best_thresholds


def predict(models, X_seq):
    result = []
    for one_model in models:
        one_result = []
        for model in one_model:
            y_pred = model.predict(X_seq, batch_size=1024)
            one_result.append(evaluate_utils.to_normal_label_ndarray(y_pred))
        result.append(cv_utils.average_of_pred(one_result))
    return cv_utils.average_of_pred(result)


def bigger_num(threshold, nums):
    result = 0
    for i, num in enumerate(nums):
        if num > threshold[i]:
            result += 1
    return result


def classify_sentence(models, sentence, single_or_total='single'):
    sentence = prepare_data_for_nn_subject.sentence_to_word_seq(sentence)
    X_seq = tokenizer.texts_to_sequences([sentence])
    X_seq = sequence.pad_sequences(X_seq, maxlen=maxlen)
    y_pred = predict(models, X_seq)
    threshold = thresholds.topic_positive_threshold
    num = bigger_num(threshold, y_pred[0])
    if num == 0:
        return ''
    predict_subject = result_utils.convert_subject_predict(y_pred, threshold)
    if num > 1 and single_or_total == 'single':
        print("一个句子中包含一个以上主题:%s, %s" % (sentence, predict_subject[0]))
    return predict_subject[0]


def classify_sentences(models, sentences):
    result = []
    for sentence in sentences:
        result.append(classify_sentence(models, sentence))
    return result


def split_sample(data, models):
    result = []
    exact_count = 0
    for i in range(len(data)):
        sample = data[i]
        parts = sample.split(',')
        text = parts[1]

        total_predict_subject = classify_sentence(models, text, single_or_total='total')
        if '|' not in total_predict_subject:
            result.append(sample)
            continue

        subject = parts[2]

        if len(re.findall('[^，。？！；…]+[，。？！；…]$', text)) == 0:
            text += '。'
        sub_sentences = re.findall('[^，。？！；…]+[，。？！；…]', text)
        if len(sub_sentences) == 1:
            result.append(sample)
            continue
        finds = []
        predict_sentences_result = classify_sentences(models, sub_sentences)
        for j in range(len(predict_sentences_result)):
            if subject in predict_sentences_result[j]:
                finds.append(sub_sentences[j])

        if len(finds) > 0:
            exact_text = ' '.join(finds)
            if len(exact_text) < 4:
                print('长度小于4：' + exact_text)
                result.append(sample)
            else:
                parts[1] = exact_text
                result.append(','.join(parts))
                exact_count += 1
        else:
            result.append(sample)
    print('exact_count: %d' % exact_count)
    return result

print('train')
train_sample = head + split_sample(train_data, models)
file_utils.write_lines(train_sample, data_path.train_sentiment_value_exact_file_path)

print('val')
val_sample = head + split_sample(val_data, models)
file_utils.write_lines(val_sample, data_path.val_sentiment_value_exact_file_path)

print('test')
test_sample = head + split_sample(test_data, models)
file_utils.write_lines(test_sample, data_path.test_public_for_sentiment_value_exact_file_path)
