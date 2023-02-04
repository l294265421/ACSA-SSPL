"""
16年的训练集就是由15年的训练集和测试集加在一起组成的，所以不考虑15年的数据
"""

from xml.dom.minidom import parse
import xml.dom.minidom

from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import file_utils

unique_example = set()
drop_by_unique_count = 0

def get_all_example_from_xml_2014(file_path):
    """

    :param file_path: str，xml文件路径，文件内容格式如下：
    <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <sentences>
        <sentence id="3121">
            <text>But the staff was so horrible to us.</text>
            <aspectTerms>
                <aspectTerm term="staff" polarity="negative" from="8" to="13"/>
            </aspectTerms>
            <aspectCategories>
                <aspectCategory category="service" polarity="negative"/>
            </aspectCategories>
        </sentence>
    </sentences>
    :return: list of str，每个str表示一个example，example示例如下：
    3121\tBut the staff was so horrible to us.\tservice\t-1
    """
    result = []
    DOMTree = xml.dom.minidom.parse(file_path)
    collection = DOMTree.documentElement
    sentences = collection.getElementsByTagName("sentence")
    sentiment_count = {'positive': 0,
                       'negative': 0,
                       'neutral': 0}
    for sentence in sentences:
        sentence_id = sentence.getAttribute("id")
        text = sentence.getElementsByTagName('text')[0].childNodes[0].data
        text = text.replace('\t', ' ')
        aspect_categories = sentence.getElementsByTagName('aspectCategory')
        for category in aspect_categories:
            aspect = category.getAttribute("category")
            sentiment = category.getAttribute('polarity')
            if sentiment == 'conflict':
                sentiment = 'neutral'
            sentiment_count[sentiment] += 1
            result.append('\t'.join([sentence_id, text, aspect, sentiment]))
    print(sentiment_count)
    return result


def get_all_example_from_xml_2016(file_path):
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
    global drop_by_unique_count
    global unique_example
    result = []
    DOMTree = xml.dom.minidom.parse(file_path)
    collection = DOMTree.documentElement
    sentences = collection.getElementsByTagName("sentence")
    sentiment_count = {'positive': 0,
                       'negative': 0,
                       'neutral': 0,
                       }
    category_count = {}
    sentence_without_opinion_count = 0
    for sentence in sentences:
        sentence_id = sentence.getAttribute("id")
        text = sentence.getElementsByTagName('text')[0].childNodes[0].data
        text = text.replace('\t', ' ')
        opinions = sentence.getElementsByTagName('Opinion')
        if len(opinions) == 0:
            sentence_without_opinion_count += 1
            # print('%s do not have any opinion' % text)
            continue

        category_and_score = {}
        sentiment_and_score = {'positive': 1,
                               'negative': -1,
                               'neutral': 0,
                               }
        for opinion in opinions:
            category = opinion.getAttribute("category").split('#')[0]
            if category not in category_and_score:
                category_and_score[category] = 0
            sentiment = opinion.getAttribute('polarity')
            category_and_score[category] += sentiment_and_score[sentiment]

        for category, score in category_and_score.items():
            if category in category_count:
                category_count[category] += 1
            else:
                category_count[category] = 1
            if score > 0:
                sentiment = 'positive'
            elif score < 0:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            example = '\t'.join([sentence_id, text, category, sentiment])
            example_without_id = '\t'.join([text, category, sentiment])
            if example not in unique_example:
                result.append(example)
                unique_example.add(example)
            else:
                drop_by_unique_count += 1
                continue
            sentiment_count[sentiment] += 1

    print('sentence_without_opinion_count: %d' % sentence_without_opinion_count)
    print(sentiment_count)
    # print(category_count)
    return result

head = ['content_id\tcontent\tsubject\tsentiment_value']
train_xml_file_path = data_path.data_base_dir + '14/Restaurants_Train_v2.xml'
train_example_2014 = get_all_example_from_xml_2014(train_xml_file_path)
train_xml_file_path = data_path.data_base_dir + '15/ABSA-15_Restaurants_Train_Final.xml'
train_example_2015 = get_all_example_from_xml_2016(train_xml_file_path)
train_xml_file_path = data_path.data_base_dir + '16/ABSA16_Restaurants_Train_SB1_v2.xml'
train_example_2016 = get_all_example_from_xml_2016(train_xml_file_path)

train_example = []
file_utils.write_lines(head + train_example, data_path.data_base_dir + '/train.csv')

test_xml_file_path = data_path.data_base_dir + '14/Restaurants_Test_Gold.xml'
text_examples_2014 = get_all_example_from_xml_2014(test_xml_file_path)
test_xml_file_path = data_path.data_base_dir + '15/ABSA15_Restaurants_Test.xml'
text_examples_2015 = get_all_example_from_xml_2016(test_xml_file_path)
test_xml_file_path = data_path.data_base_dir + '16/EN_REST_SB1_TEST.xml.gold'
text_examples_2016 = get_all_example_from_xml_2016(test_xml_file_path)

text_examples = []
file_utils.write_lines(head + text_examples, data_path.data_base_dir + '/test_public.csv')
file_utils.write_lines(head + text_examples, data_path.data_base_dir + '/test_public_gold.csv')

print(drop_by_unique_count)

# {'positive': 2179, 'negative': 839, 'neutral': 695}
# {'positive': 981, 'negative': 309, 'neutral': 75}
# {'RESTAURANT': 367, 'SERVICE': 256, 'FOOD': 503, 'DRINKS': 57, 'AMBIENCE': 162, 'LOCATION': 20}
# {'positive': 1353, 'negative': 596, 'neutral': 124}
# {'RESTAURANT': 564, 'SERVICE': 419, 'FOOD': 757, 'DRINKS': 79, 'AMBIENCE': 226, 'LOCATION': 28}
# {'positive': 657, 'negative': 222, 'neutral': 146}
# {'positive': 372, 'negative': 289, 'neutral': 46}
# {'RESTAURANT': 196, 'FOOD': 254, 'AMBIENCE': 69, 'DRINKS': 21, 'SERVICE': 159, 'LOCATION': 8}
# {'positive': 482, 'negative': 169, 'neutral': 49}
# {'FOOD': 257, 'RESTAURANT': 193, 'SERVICE': 145, 'AMBIENCE': 57, 'DRINKS': 35, 'LOCATION': 13}
