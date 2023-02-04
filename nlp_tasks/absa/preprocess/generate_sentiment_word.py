from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import data_utils
from nlp_tasks.absa.utils import file_utils

sentiment_word = data_utils.read_field(data_path.train_original_file_path, 4)[1:]
sentiment_word = [word for word in sentiment_word if len(word) != 0]
sentiment_word_uniq = list(set(sentiment_word))
file_utils.write_lines(sentiment_word_uniq, data_path.sentiment_word_file_path)
