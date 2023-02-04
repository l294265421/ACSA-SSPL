from xml.dom.minidom import parse
import xml.dom.minidom

from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import file_utils

category_count_totle = {}


def get_all_example_from_xml(file_path):
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
            # print('%s do not have any opinion'  % text)
            continue
        for opinion in opinions:
            category = opinion.getAttribute("category")
            if category in category_count:
                category_count[category] += 1
            else:
                category_count[category] = 1
            sentiment = opinion.getAttribute('polarity')
            sentiment_count[sentiment] += 1
            result.append('\t'.join([sentence_id, text, category, sentiment]))
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
train_xml_file_path = data_path.data_base_dir + 'ABSA16_Laptops_Train_SB1_v2.xml'
train_example = get_all_example_from_xml(train_xml_file_path)
file_utils.write_lines(head + train_example, data_path.data_base_dir + '/train.csv')

test_xml_file_path = data_path.data_base_dir + 'EN_LAPT_SB1_TEST_.xml.gold'
text_examples = get_all_example_from_xml(test_xml_file_path)
file_utils.write_lines(head + text_examples, data_path.data_base_dir + '/test_public.csv')
file_utils.write_lines(head + text_examples, data_path.data_base_dir + '/test_public_gold.csv')

category_index = {}
for c in category_count_totle.keys():
    category_index[c] = str(len(category_index))
print(category_index)

#sentence_without_opinion_count: 292
# {'positive': 1657, 'negative': 749, 'neutral': 101}
# {'RESTAURANT#GENERAL': 422, 'SERVICE#GENERAL': 449, 'FOOD#QUALITY': 849, 'FOOD#STYLE_OPTIONS': 137, 'DRINKS#STYLE_OPTIONS': 32, 'DRINKS#PRICES': 20, 'RESTAURANT#PRICES': 80, 'RESTAURANT#MISCELLANEOUS': 98, 'AMBIENCE#GENERAL': 255, 'FOOD#PRICES': 90, 'LOCATION#GENERAL': 28, 'DRINKS#QUALITY': 47}
# sentence_without_opinion_count: 89
# {'positive': 611, 'negative': 204, 'neutral': 44}
# {'FOOD#QUALITY': 313, 'FOOD#STYLE_OPTIONS': 55, 'RESTAURANT#GENERAL': 142, 'SERVICE#GENERAL': 155, 'AMBIENCE#GENERAL': 66, 'DRINKS#STYLE_OPTIONS': 12, 'FOOD#PRICES': 23, 'RESTAURANT#PRICES': 21, 'LOCATION#GENERAL': 13, 'DRINKS#QUALITY': 22, 'RESTAURANT#MISCELLANEOUS': 33, 'DRINKS#PRICES': 4}
# {'RESTAURANT#GENERAL': '0', 'SERVICE#GENERAL': '1', 'FOOD#QUALITY': '2', 'FOOD#STYLE_OPTIONS': '3', 'DRINKS#STYLE_OPTIONS': '4', 'DRINKS#PRICES': '5', 'RESTAURANT#PRICES': '6', 'RESTAURANT#MISCELLANEOUS': '7', 'AMBIENCE#GENERAL': '8', 'FOOD#PRICES': '9', 'LOCATION#GENERAL': '10', 'DRINKS#QUALITY': '11'}