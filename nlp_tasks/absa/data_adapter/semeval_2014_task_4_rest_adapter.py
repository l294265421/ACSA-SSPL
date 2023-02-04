from xml.dom.minidom import parse
import xml.dom.minidom

from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import file_utils


def get_all_example_from_xml(file_path):
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
                       'neutral': 0,
                       'conflict': 0}
    for sentence in sentences:
        sentence_id = sentence.getAttribute("id")
        text = sentence.getElementsByTagName('text')[0].childNodes[0].data
        text = text.replace('\t', ' ')
        aspect_categories = sentence.getElementsByTagName('aspectCategory')
        for category in aspect_categories:
            aspect = category.getAttribute("category")
            if '/' in aspect:
                aspect = aspect.split('/')[1]
            sentiment = category.getAttribute('polarity')
            sentiment_count[sentiment] += 1
            result.append('\t'.join([sentence_id, text, aspect, sentiment]))
    print(sentiment_count)
    return result


head = ['content_id\tcontent\tsubject\tsentiment_value']
train_xml_file_path = data_path.data_base_dir + 'Restaurants_Train_v2.xml'
train_example = get_all_example_from_xml(train_xml_file_path)
file_utils.write_lines(head + train_example, data_path.data_base_dir + '/train.csv')

test_xml_file_path = data_path.data_base_dir + 'Restaurants_Test_Gold.xml'
text_examples = get_all_example_from_xml(test_xml_file_path)
file_utils.write_lines(head + text_examples, data_path.data_base_dir + '/test_public.csv')
file_utils.write_lines(head + text_examples, data_path.data_base_dir + '/test_public_gold.csv')

# {'positive': 2179, 'negative': 839, 'neutral': 500, 'conflict': 195}
# {'positive': 657, 'negative': 222, 'neutral': 94, 'conflict': 52}
