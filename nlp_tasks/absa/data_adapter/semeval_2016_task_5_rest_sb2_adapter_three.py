from xml.dom.minidom import parse
import xml.dom.minidom

from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import file_utils

category_count_totle = {}


def get_all_example_from_xml(file_path, category_count={}):
    """

    :param file_path: str，xml文件路径，文件内容格式如下：
    <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <Reviews>
        <Review rid="1004293">
            <sentences>
                <sentence id="1004293:0">
                    <text>Judging from previous posts this used to be a good place, but not any longer.</text>
                    <Opinions>
                        <Opinion target="place" category="RESTAURANT#GENERAL" polarity="negative" from="51" to="56"/>
                    </Opinions>
                </sentence>
                <sentence id="1004293:1">
                    <text>We, there were four of us, arrived at noon - the place was empty - and the staff acted like we were imposing on them and they were very rude.</text>
                    <Opinions>
                        <Opinion target="staff" category="SERVICE#GENERAL" polarity="negative" from="75" to="80"/>
                    </Opinions>
                </sentence>
            </sentences>
        </Review>
    </Reviews>
    :return: list of str，每个str表示一个example，example示例如下：
    3121\tBut the staff was so horrible to us.\tservice\t-1
    """
    result = []
    DOMTree = xml.dom.minidom.parse(file_path)
    collection = DOMTree.documentElement
    sentences = collection.getElementsByTagName("Review")
    sentiment_count = {'positive': 0,
                       'negative': 0,
                       'neutral': 0,
                       'conflict': 0
                       }
    sentence_without_opinion_count = 0
    for sentence in sentences:
        review_id = sentence.getAttribute("rid")
        texts = sentence.getElementsByTagName('text')
        texts = [e.childNodes[0].data for e in texts]
        text = ' '.join(texts)
        text = text.replace('\t', ' ')

        opinions = sentence.getElementsByTagName('Opinion')
        if len(opinions) == 0:
            sentence_without_opinion_count += 1
            # print('%s do not have any opinion'  % text)
            continue
        for opinion in opinions:
            category = opinion.getAttribute("category")
            if category in category_count:
                category_count[category] += 1
            else:
                category_count[category] = 1
            sentiment = opinion.getAttribute('polarity')
            if sentiment == 'conflict':
                continue
            sentiment_count[sentiment] += 1
            result.append('\t'.join([review_id, text, category, sentiment]))
    print('sentence_without_opinion_count: %d' % sentence_without_opinion_count)
    print(sentiment_count)
    print(category_count)
    global category_count_totle
    for k, v in category_count.items():
        if k in category_count_totle:
            category_count_totle[k] += v
        else:
            category_count_totle[k] = v

    return result


head = ['content_id\tcontent\tsubject\tsentiment_value']
train_xml_file_path = data_path.data_base_dir + 'ABSA16_Restaurants_Train_English_SB2.xml'
train_category_count = {}
train_example = get_all_example_from_xml(train_xml_file_path, category_count=train_category_count)
file_utils.write_lines(head + train_example, data_path.data_base_dir + '/train.csv')

test_xml_file_path = data_path.data_base_dir + 'EN_REST_SB2_TEST.xml.gold'
test_examples = get_all_example_from_xml(test_xml_file_path)
file_utils.write_lines(head + test_examples, data_path.data_base_dir + '/test_public.csv')
file_utils.write_lines(head + test_examples, data_path.data_base_dir + '/test_public_gold.csv')

q1_test_data = []
q2_test_data = []
q3_test_data = []
q4_test_data = []
q_numbers = [36.5, 68.5, 177]
for example in test_examples:
    categoty = example.split('\t')[2]
    if categoty not in train_category_count or train_category_count[categoty] <= q_numbers[0]:
        q1_test_data.append(example)
    elif train_category_count[categoty] <= q_numbers[1]:
        q2_test_data.append(example)
    elif train_category_count[categoty] <= q_numbers[2]:
        q3_test_data.append(example)
    else:
        q4_test_data.append(example)
file_utils.write_lines(head + q1_test_data, data_path.data_base_dir + '/q1_test_data.csv')
file_utils.write_lines(head + q2_test_data, data_path.data_base_dir + '/q2_test_data.csv')
file_utils.write_lines(head + q3_test_data, data_path.data_base_dir + '/q3_test_data.csv')
file_utils.write_lines(head + q4_test_data, data_path.data_base_dir + '/q4_test_data.csv')

category_index = {}
for c in category_count_totle.keys():
    category_index[c] = str(len(category_index))
print(category_index)

# 训练集各个类别的样本相对较多，参数共享对情感的影响较小
# sentence_without_opinion_count: 0
# {'positive': 1012, 'negative': 327, 'neutral': 55, 'conflict': 41}
# {'RESTAURANT#GENERAL': 335, 'SERVICE#GENERAL': 213, 'FOOD#QUALITY': 314, 'FOOD#STYLE_OPTIONS': 95, 'DRINKS#STYLE_OPTIONS': 29, 'DRINKS#PRICES': 20, 'RESTAURANT#PRICES': 63, 'AMBIENCE#GENERAL': 165, 'RESTAURANT#MISCELLANEOUS': 68, 'LOCATION#GENERAL': 25, 'FOOD#PRICES': 69, 'DRINKS#QUALITY': 39}
# sentence_without_opinion_count: 0
# {'positive': 286, 'negative': 84, 'neutral': 23, 'conflict': 11}
# {'FOOD#QUALITY': 86, 'FOOD#STYLE_OPTIONS': 32, 'RESTAURANT#GENERAL': 90, 'SERVICE#GENERAL': 64, 'AMBIENCE#GENERAL': 38, 'RESTAURANT#PRICES': 16, 'DRINKS#STYLE_OPTIONS': 11, 'FOOD#PRICES': 18, 'LOCATION#GENERAL': 9, 'DRINKS#QUALITY': 15, 'RESTAURANT#MISCELLANEOUS': 24, 'DRINKS#PRICES': 1}
# {'RESTAURANT#GENERAL': '0', 'SERVICE#GENERAL': '1', 'FOOD#QUALITY': '2', 'FOOD#STYLE_OPTIONS': '3', 'DRINKS#STYLE_OPTIONS': '4', 'DRINKS#PRICES': '5', 'RESTAURANT#PRICES': '6', 'AMBIENCE#GENERAL': '7', 'RESTAURANT#MISCELLANEOUS': '8', 'LOCATION#GENERAL': '9', 'FOOD#PRICES': '10', 'DRINKS#QUALITY': '11'}