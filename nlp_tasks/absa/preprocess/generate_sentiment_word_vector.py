from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import file_utils
from nlp_tasks.absa.utils import embedding_utils

sentiment_words = set()
sentiment_dict = file_utils.read_all_lines(data_path.sentiment_dict_file_path)[1:]
for element in sentiment_dict:
    sentiment_words.add(element.split(',')[0])

sentiment_word = file_utils.read_all_lines(data_path.sentiment_word_file_path)
for word in sentiment_word:
    sentiment_words.add(word)

embedding_all = embedding_utils.generate_word_embedding_all(data_path.embedding_file)

result = []
for key, value in embedding_all.items():
    value = value.tolist()
    if len(value) == 1:
        continue
    if key in sentiment_words:
        print(key)
        value.append(1)
    else:
        value.append(0)
    element = [key] + value
    element = [str(e) for e in element]
    result.append(' '.join(element))

file_utils.write_lines(result, data_path.embedding_sentiment_filepath)
