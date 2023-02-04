from xml.dom.minidom import parse
import xml.dom.minidom

from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import file_utils


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
            # print('%s do not have any opinion' % text)
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
    return result


head = ['content_id\tcontent\tsubject\tsentiment_value']
train_xml_file_path = data_path.data_base_dir + 'ABSA-15_Restaurants_Train_Final.xml'
train_example = get_all_example_from_xml(train_xml_file_path)
file_utils.write_lines(head + train_example, data_path.data_base_dir + '/train.csv')

test_xml_file_path = data_path.data_base_dir + 'ABSA15_Restaurants_Test.xml'
text_examples = get_all_example_from_xml(test_xml_file_path)
file_utils.write_lines(head + text_examples, data_path.data_base_dir + '/test_public.csv')
file_utils.write_lines(head + text_examples, data_path.data_base_dir + '/test_public_gold.csv')

# sentence_without_opinion_count: 195
# {'positive': 1198, 'negative': 403, 'neutral': 53}
# {'RESTAURANT#GENERAL': 269, 'SERVICE#GENERAL': 268, 'FOOD#QUALITY': 581, 'FOOD#STYLE_OPTIONS': 93, 'DRINKS#STYLE_OPTIONS': 26, 'DRINKS#PRICES': 15, 'RESTAURANT#PRICES': 48, 'AMBIENCE#GENERAL': 183, 'RESTAURANT#MISCELLANEOUS': 62, 'FOOD#PRICES': 54, 'LOCATION#GENERAL': 20, 'DRINKS#QUALITY': 34, 'FOOD#GENERAL': 1}
# sentence_without_opinion_count: 103
# {'positive': 454, 'negative': 346, 'neutral': 45}
# {'RESTAURANT#GENERAL': 147, 'FOOD#QUALITY': 271, 'FOOD#PRICES': 31, 'RESTAURANT#MISCELLANEOUS': 38, 'AMBIENCE#GENERAL': 77, 'FOOD#STYLE_OPTIONS': 40, 'DRINKS#QUALITY': 12, 'SERVICE#GENERAL': 175, 'LOCATION#GENERAL': 8, 'RESTAURANT#PRICES': 35, 'DRINKS#STYLE_OPTIONS': 6, 'DRINKS#PRICES': 5}