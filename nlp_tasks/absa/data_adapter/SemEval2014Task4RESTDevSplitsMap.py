import re
from string import punctuation

from nlp_tasks.utils import file_utils
from nlp_tasks.absa.data_adapter.data_object import Semeval2014Task4RestDevSplits, Semeval2014Task4Rest


def clean_str(sentence: str):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """

    sentence_clean = sentence.lower()
    sentence_clean = re.sub('[!"#$%&\'\-()*+,./:;<=>?@[\\]^_`{|}~]', '', sentence_clean)
    sentence_clean = sentence_clean.strip()
    sentence_clean = re.sub('\s', '', sentence_clean)
    return sentence_clean


data_type_and_data_original = Semeval2014Task4Rest().get_data_type_and_data_dict()
data_type_and_data = Semeval2014Task4RestDevSplits().get_data_type_and_data_dict()

sentences_original_mapping = {}
sentences_original_mapping_output = []
for data_type, data in data_type_and_data_original.items():
    if data is None:
        continue
    for sample in data:
        sentence = sample.absa_sentences[0].text
        sentence_clean = clean_str(sentence)
        sentences_original_mapping[sentence_clean] = sentence
        sentences_original_mapping_output.append('%s\t%s' % (sentence, sentence_clean))
file_utils.write_lines(sentences_original_mapping_output, 'd:/sentences_original_mapping_output.txt')

sentence_map = []
for data_type, data in data_type_and_data.items():
    if data is None:
        continue
    for sample in data:
        sentence = sample.absa_sentences[0].text
        sentence_clean = re.sub('\s', '', sentence)
        try:
            sentence_original = sentences_original_mapping[sentence_clean]
            sentence_map.append('%s\t%s' % (sentence, sentence_original.strip()))
        except:
            print(sentence)
file_utils.write_lines(sentence_map, 'd:/sentence_map.txt')





