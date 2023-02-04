import json
import sys

from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import embedding_utils

word_index_sentiment = json.load(open(data_path.data_base_dir + data_path.word_index_sentiment_file_path, encoding='utf-8'))
words = word_index_sentiment.keys()

embedding_all = embedding_utils.generate_word_embedding_all(data_path.embedding_base_dir + sys
                                                            .argv[1])

words_no_embedding = [word for word in words if word not in embedding_all]

print(len(words_no_embedding))
for word in words_no_embedding:
    print(word)
