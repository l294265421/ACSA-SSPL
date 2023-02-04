from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import embedding_utils
from nlp_tasks.absa.utils import file_utils

aspect_categories = ['food', 'service', 'price', 'ambience', 'miscellaneous']

embedding_all = embedding_utils.generate_word_embedding_all(data_path.embedding_file)

no_embedding_aspect_categories = []
for aspect_category in aspect_categories:
    if aspect_category not in embedding_all:
        no_embedding_aspect_categories.append(aspect_category)
    else:
        print(aspect_category)
        print(embedding_all[aspect_category])
print(no_embedding_aspect_categories)

result = []
for key, value in embedding_all.items():
    value = value.tolist()
    element = [key] + value
    element = [str(e) for e in element]
    result.append(' '.join(element))

for aspect_category in no_embedding_aspect_categories:
    vector = result[0].split(' ')
    vector[0] = aspect_category
    result.append(' '.join(vector))

file_utils.write_lines(result, data_path.embedding_sentiment_filepath)
