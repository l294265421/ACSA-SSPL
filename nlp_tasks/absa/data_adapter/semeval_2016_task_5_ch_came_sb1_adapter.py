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
    sentences = collection.getElementsByTagName("sentence")
    sentiment_count = {'positive': 0,
                       'negative': 0,
                       'neutral': 0,
                       }
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
train_xml_file_path = data_path.data_base_dir + 'camera_training.xml'
train_category_count = {}
train_example = get_all_example_from_xml(train_xml_file_path, category_count=train_category_count)
file_utils.write_lines(head + train_example, data_path.data_base_dir + '/train.csv')

test_xml_file_path = data_path.data_base_dir + 'CH_CAME_SB1_TEST_.xml'
test_examples = get_all_example_from_xml(test_xml_file_path)
file_utils.write_lines(head + test_examples, data_path.data_base_dir + '/test_public.csv')
file_utils.write_lines(head + test_examples, data_path.data_base_dir + '/test_public_gold.csv')

q1_test_data = []
q2_test_data = []
q3_test_data = []
q4_test_data = []
q_numbers = [1, 6, 13]
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

# sentence_without_opinion_count: 4519
# {'positive': 809, 'negative': 450, 'neutral': 0}
# {'CAMERA#DESIGN_FEATURES': 77, 'CAMERA#OPERATION_PERFORMANCE': 137, 'HARDWARE#USABILITY': 9, 'SOFTWARE#DESIGN_FEATURES': 20, 'LENS#GENERAL': 14, 'LENS#OPERATION_PERFORMANCE': 28, 'BATTERY#QUALITY': 5, 'CAMERA#PORTABILITY': 30, 'PHOTO#QUALITY': 259, 'CAMERA#QUALITY': 88, 'FOCUS#OPERATION_PERFORMANCE': 140, 'HARDWARE#DESIGN_FEATURES': 45, 'SOFTWARE#OPERATION_PERFORMANCE': 19, 'LENS#DESIGN_FEATURES': 22, 'CAMERA#GENERAL': 12, 'HARDWARE#GENERAL': 7, 'HARDWARE#OPERATION_PERFORMANCE': 55, 'DISPLAY#DESIGN_FEATURES': 11, 'DISPLAY#OPERATION_PERFORMANCE': 12, 'CAMERA#PRICE': 24, 'DISPLAY#USABILITY': 6, 'PHOTO#GENERAL': 37, 'HARDWARE#QUALITY': 8, 'KEYBOARD#DESIGN_FEATURES': 14, 'HARDWARE#PRICE': 1, 'FOCUS#GENERAL': 8, 'SOFTWARE#QUALITY': 4, 'FOCUS#QUALITY': 8, 'FOCUS#DESIGN_FEATURES': 7, 'LENS#QUALITY': 13, 'FOCUS#USABILITY': 4, 'BATTERY#DESIGN_FEATURES': 1, 'CAMERA#USABILITY': 7, 'OS#OPERATION_PERFORMANCE': 14, 'MEMORY#OPERATION_PERFORMANCE': 4, 'PORTS#OPERATION_PERFORMANCE': 1, 'BATTERY#USABILITY': 3, 'KEYBOARD#OPERATION_PERFORMANCE': 3, 'DISPLAY#QUALITY': 13, 'LENS#PORTABILITY': 1, 'MULTIMEDIA_DEVICES#CONNECTIVITY': 4, 'MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE': 7, 'MULTIMEDIA_DEVICES#DESIGN_FEATURES': 1, 'MULTIMEDIA_DEVICES#USABILITY': 2, 'SOFTWARE#USABILITY': 6, 'OS#GENERAL': 2, 'PORTS#CONNECTIVITY': 1, 'DISPLAY#GENERAL': 8, 'PORTS#QUALITY': 1, 'CAMERA#MISCELLANEOUS': 7, 'MULTIMEDIA_DEVICES#QUALITY': 4, 'POWER_SUPPLY#USABILITY': 1, 'SOFTWARE#GENERAL': 11, 'POWER_SUPPLY#OPERATION_PERFORMANCE': 1, 'LENS#MISCELLANEOUS': 1, 'PHOTO#MISCELLANEOUS': 2, 'SOFTWARE#MISCELLANEOUS': 2, 'HARDWARE#MISCELLANEOUS': 3, 'FOCUS#MISCELLANEOUS': 6, 'BATTERY#OPERATION_PERFORMANCE': 9, 'MULTIMEDIA_DEVICES#GENERAL': 1, 'OS#QUALITY': 1, 'SOFTWARE#CONNECTIVITY': 1, 'PORTS#DESIGN_FEATURES': 1, 'LENS#PRICE': 1, 'LENS#USABILITY': 1, 'KEYBOARD#QUALITY': 1, 'CPU#OPERATION_PERFORMANCE': 1, 'DISPLAY#MISCELLANEOUS': 1}
# sentence_without_opinion_count: 1775
# {'positive': 344, 'negative': 137, 'neutral': 0}
# {'CAMERA#QUALITY': 29, 'FOCUS#OPERATION_PERFORMANCE': 30, 'HARDWARE#QUALITY': 6, 'CAMERA#OPERATION_PERFORMANCE': 56, 'CAMERA#DESIGN_FEATURES': 48, 'CAMERA#USABILITY': 8, 'SOFTWARE#DESIGN_FEATURES': 10, 'HARDWARE#OPERATION_PERFORMANCE': 12, 'PHOTO#QUALITY': 118, 'LENS#OPERATION_PERFORMANCE': 9, 'DISPLAY#QUALITY': 2, 'BATTERY#QUALITY': 5, 'FOCUS#QUALITY': 4, 'MULTIMEDIA_DEVICES#CONNECTIVITY': 4, 'CAMERA#PRICE': 8, 'HARDWARE#DESIGN_FEATURES': 16, 'HARDWARE#USABILITY': 5, 'LENS#QUALITY': 7, 'LENS#MISCELLANEOUS': 1, 'SOFTWARE#MISCELLANEOUS': 1, 'PHOTO#MISCELLANEOUS': 2, 'FOCUS#DESIGN_FEATURES': 4, 'CAMERA#GENERAL': 5, 'LENS#GENERAL': 1, 'PHOTO#GENERAL': 6, 'SOFTWARE#QUALITY': 6, 'SOFTWARE#OPERATION_PERFORMANCE': 13, 'SOFTWARE#GENERAL': 2, 'SOFTWARE#USABILITY': 4, 'KEYBOARD#DESIGN_FEATURES': 6, 'HARDWARE#GENERAL': 4, 'OS#OPERATION_PERFORMANCE': 7, 'DISPLAY#OPERATION_PERFORMANCE': 4, 'CAMERA#MISCELLANEOUS': 1, 'CAMERA#PORTABILITY': 2, 'DISPLAY#DESIGN_FEATURES': 6, 'LENS#DESIGN_FEATURES': 7, 'MEMORY#QUALITY': 1, 'MEMORY#OPERATION_PERFORMANCE': 1, 'FOCUS#GENERAL': 3, 'PORTS#DESIGN_FEATURES': 1, 'MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE': 4, 'HARDWARE#CONNECTIVITY': 1, 'MULTIMEDIA_DEVICES#QUALITY': 3, 'BATTERY#OPERATION_PERFORMANCE': 1, 'OS#USABILITY': 1, 'MULTIMEDIA_DEVICES#GENERAL': 1, 'MEMORY#DESIGN_FEATURES': 2, 'DISPLAY#USABILITY': 1, 'SUPPORT#GENERAL': 1, 'OS#DESIGN_FEATURES': 1}
# {'CAMERA#DESIGN_FEATURES': '0', 'CAMERA#OPERATION_PERFORMANCE': '1', 'HARDWARE#USABILITY': '2', 'SOFTWARE#DESIGN_FEATURES': '3', 'LENS#GENERAL': '4', 'LENS#OPERATION_PERFORMANCE': '5', 'BATTERY#QUALITY': '6', 'CAMERA#PORTABILITY': '7', 'PHOTO#QUALITY': '8', 'CAMERA#QUALITY': '9', 'FOCUS#OPERATION_PERFORMANCE': '10', 'HARDWARE#DESIGN_FEATURES': '11', 'SOFTWARE#OPERATION_PERFORMANCE': '12', 'LENS#DESIGN_FEATURES': '13', 'CAMERA#GENERAL': '14', 'HARDWARE#GENERAL': '15', 'HARDWARE#OPERATION_PERFORMANCE': '16', 'DISPLAY#DESIGN_FEATURES': '17', 'DISPLAY#OPERATION_PERFORMANCE': '18', 'CAMERA#PRICE': '19', 'DISPLAY#USABILITY': '20', 'PHOTO#GENERAL': '21', 'HARDWARE#QUALITY': '22', 'KEYBOARD#DESIGN_FEATURES': '23', 'HARDWARE#PRICE': '24', 'FOCUS#GENERAL': '25', 'SOFTWARE#QUALITY': '26', 'FOCUS#QUALITY': '27', 'FOCUS#DESIGN_FEATURES': '28', 'LENS#QUALITY': '29', 'FOCUS#USABILITY': '30', 'BATTERY#DESIGN_FEATURES': '31', 'CAMERA#USABILITY': '32', 'OS#OPERATION_PERFORMANCE': '33', 'MEMORY#OPERATION_PERFORMANCE': '34', 'PORTS#OPERATION_PERFORMANCE': '35', 'BATTERY#USABILITY': '36', 'KEYBOARD#OPERATION_PERFORMANCE': '37', 'DISPLAY#QUALITY': '38', 'LENS#PORTABILITY': '39', 'MULTIMEDIA_DEVICES#CONNECTIVITY': '40', 'MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE': '41', 'MULTIMEDIA_DEVICES#DESIGN_FEATURES': '42', 'MULTIMEDIA_DEVICES#USABILITY': '43', 'SOFTWARE#USABILITY': '44', 'OS#GENERAL': '45', 'PORTS#CONNECTIVITY': '46', 'DISPLAY#GENERAL': '47', 'PORTS#QUALITY': '48', 'CAMERA#MISCELLANEOUS': '49', 'MULTIMEDIA_DEVICES#QUALITY': '50', 'POWER_SUPPLY#USABILITY': '51', 'SOFTWARE#GENERAL': '52', 'POWER_SUPPLY#OPERATION_PERFORMANCE': '53', 'LENS#MISCELLANEOUS': '54', 'PHOTO#MISCELLANEOUS': '55', 'SOFTWARE#MISCELLANEOUS': '56', 'HARDWARE#MISCELLANEOUS': '57', 'FOCUS#MISCELLANEOUS': '58', 'BATTERY#OPERATION_PERFORMANCE': '59', 'MULTIMEDIA_DEVICES#GENERAL': '60', 'OS#QUALITY': '61', 'SOFTWARE#CONNECTIVITY': '62', 'PORTS#DESIGN_FEATURES': '63', 'LENS#PRICE': '64', 'LENS#USABILITY': '65', 'KEYBOARD#QUALITY': '66', 'CPU#OPERATION_PERFORMANCE': '67', 'DISPLAY#MISCELLANEOUS': '68', 'MEMORY#QUALITY': '69', 'HARDWARE#CONNECTIVITY': '70', 'OS#USABILITY': '71', 'MEMORY#DESIGN_FEATURES': '72', 'SUPPORT#GENERAL': '73', 'OS#DESIGN_FEATURES': '74'}

