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
train_xml_file_path = data_path.data_base_dir + 'ABSA16_Laptops_Train_English_SB2.xml'
train_category_count = {}
train_example = get_all_example_from_xml(train_xml_file_path, category_count=train_category_count)
file_utils.write_lines(head + train_example, data_path.data_base_dir + '/train.csv')

test_xml_file_path = data_path.data_base_dir + 'EN_LAPT_SB2_TEST.xml.gold'
test_examples = get_all_example_from_xml(test_xml_file_path)
file_utils.write_lines(head + test_examples, data_path.data_base_dir + '/test_public.csv')
file_utils.write_lines(head + test_examples, data_path.data_base_dir + '/test_public_gold.csv')

q1_test_data = []
q2_test_data = []
q3_test_data = []
q4_test_data = []
q_numbers = [2, 7, 20]
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

# {'positive': 1210, 'negative': 708, 'neutral': 123, 'conflict': 41}
# {'LAPTOP#GENERAL': 395, 'LAPTOP#OPERATION_PERFORMANCE': 196, 'HARD_DISC#DESIGN_FEATURES': 12, 'LAPTOP#QUALITY': 152, 'DISPLAY#QUALITY': 47, 'LAPTOP#DESIGN_FEATURES': 154, 'KEYBOARD#GENERAL': 12, 'LAPTOP#USABILITY': 102, 'LAPTOP#PRICE': 120, 'COMPANY#GENERAL': 89, 'SUPPORT#QUALITY': 48, 'LAPTOP#CONNECTIVITY': 44, 'DISPLAY#OPERATION_PERFORMANCE': 10, 'LAPTOP#PORTABILITY': 42, 'BATTERY#OPERATION_PERFORMANCE': 67, 'MULTIMEDIA_DEVICES#GENERAL': 6, 'MULTIMEDIA_DEVICES#QUALITY': 23, 'MULTIMEDIA_DEVICES#DESIGN_FEATURES': 4, 'MULTIMEDIA_DEVICES#USABILITY': 4, 'KEYBOARD#DESIGN_FEATURES': 26, 'KEYBOARD#USABILITY': 20, 'HARD_DISC#QUALITY': 11, 'WARRANTY#GENERAL': 3, 'MOUSE#DESIGN_FEATURES': 11, 'DISPLAY#GENERAL': 22, 'LAPTOP#MISCELLANEOUS': 92, 'SOFTWARE#GENERAL': 23, 'OS#GENERAL': 33, 'GRAPHICS#QUALITY': 5, 'SHIPPING#QUALITY': 5, 'DISPLAY#DESIGN_FEATURES': 25, 'OS#OPERATION_PERFORMANCE': 9, 'DISPLAY#USABILITY': 5, 'BATTERY#QUALITY': 7, 'POWER_SUPPLY#QUALITY': 8, 'POWER_SUPPLY#OPERATION_PERFORMANCE': 2, 'HARDWARE#QUALITY': 4, 'OS#MISCELLANEOUS': 1, 'GRAPHICS#GENERAL': 15, 'MOUSE#OPERATION_PERFORMANCE': 15, 'WARRANTY#PRICE': 2, 'SHIPPING#PRICE': 2, 'MOUSE#QUALITY': 9, 'POWER_SUPPLY#DESIGN_FEATURES': 3, 'OS#USABILITY': 21, 'KEYBOARD#QUALITY': 17, 'BATTERY#MISCELLANEOUS': 1, 'KEYBOARD#OPERATION_PERFORMANCE': 7, 'MOUSE#USABILITY': 19, 'MEMORY#DESIGN_FEATURES': 17, 'PORTS#DESIGN_FEATURES': 1, 'SOFTWARE#QUALITY': 2, 'OS#QUALITY': 4, 'SOFTWARE#OPERATION_PERFORMANCE': 13, 'SOFTWARE#USABILITY': 12, 'SUPPORT#PRICE': 5, 'MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE': 2, 'MOTHERBOARD#QUALITY': 5, 'CPU#OPERATION_PERFORMANCE': 16, 'MOUSE#GENERAL': 7, 'OPTICAL_DRIVES#QUALITY': 5, 'OS#DESIGN_FEATURES': 2, 'SOFTWARE#MISCELLANEOUS': 12, 'CPU#DESIGN_FEATURES': 1, 'CPU#QUALITY': 2, 'GRAPHICS#DESIGN_FEATURES': 2, 'POWER_SUPPLY#MISCELLANEOUS': 1, 'CPU#MISCELLANEOUS': 1, 'PORTS#QUALITY': 3, 'FANS_COOLING#DESIGN_FEATURES': 1, 'OPTICAL_DRIVES#OPERATION_PERFORMANCE': 1, 'HARDWARE#OPERATION_PERFORMANCE': 1, 'SUPPORT#MISCELLANEOUS': 2, 'FANS_COOLING#QUALITY': 3, 'PORTS#OPERATION_PERFORMANCE': 1, 'GRAPHICS#MISCELLANEOUS': 1, 'FANS_COOLING#OPERATION_PERFORMANCE': 1, 'HARDWARE#GENERAL': 2, 'SOFTWARE#DESIGN_FEATURES': 4, 'SOFTWARE#PRICE': 1, 'MULTIMEDIA_DEVICES#MISCELLANEOUS': 1}
# sentence_without_opinion_count: 0
# {'positive': 338, 'negative': 162, 'neutral': 31, 'conflict': 14}
# {'LAPTOP#OPERATION_PERFORMANCE': 47, 'COMPANY#GENERAL': 24, 'LAPTOP#GENERAL': 80, 'LAPTOP#USABILITY': 30, 'LAPTOP#MISCELLANEOUS': 24, 'LAPTOP#DESIGN_FEATURES': 39, 'LAPTOP#PORTABILITY': 5, 'BATTERY#OPERATION_PERFORMANCE': 14, 'LAPTOP#PRICE': 27, 'HARD_DISC#DESIGN_FEATURES': 9, 'LAPTOP#QUALITY': 29, 'DISPLAY#QUALITY': 21, 'POWER_SUPPLY#GENERAL': 1, 'MULTIMEDIA_DEVICES#QUALITY': 4, 'SHIPPING#QUALITY': 1, 'OPTICAL_DRIVES#DESIGN_FEATURES': 1, 'MULTIMEDIA_DEVICES#GENERAL': 3, 'OS#USABILITY': 15, 'OS#GENERAL': 9, 'MOUSE#DESIGN_FEATURES': 4, 'KEYBOARD#DESIGN_FEATURES': 8, 'DISPLAY#DESIGN_FEATURES': 12, 'MOUSE#USABILITY': 9, 'POWER_SUPPLY#DESIGN_FEATURES': 2, 'HARD_DISC#GENERAL': 2, 'DISPLAY#GENERAL': 4, 'KEYBOARD#OPERATION_PERFORMANCE': 4, 'SUPPORT#QUALITY': 11, 'MEMORY#DESIGN_FEATURES': 7, 'CPU#OPERATION_PERFORMANCE': 3, 'LAPTOP#CONNECTIVITY': 7, 'KEYBOARD#USABILITY': 6, 'SOFTWARE#USABILITY': 6, 'MULTIMEDIA_DEVICES#DESIGN_FEATURES': 2, 'KEYBOARD#QUALITY': 5, 'DISPLAY#USABILITY': 5, 'DISPLAY#OPERATION_PERFORMANCE': 6, 'CPU#DESIGN_FEATURES': 3, 'GRAPHICS#DESIGN_FEATURES': 2, 'FANS_COOLING#OPERATION_PERFORMANCE': 2, 'OS#DESIGN_FEATURES': 5, 'OPTICAL_DRIVES#GENERAL': 1, 'OS#OPERATION_PERFORMANCE': 2, 'OS#MISCELLANEOUS': 2, 'BATTERY#QUALITY': 3, 'SUPPORT#PRICE': 2, 'MOTHERBOARD#QUALITY': 2, 'KEYBOARD#GENERAL': 2, 'MOUSE#GENERAL': 3, 'SOFTWARE#GENERAL': 4, 'MOUSE#OPERATION_PERFORMANCE': 2, 'HARDWARE#QUALITY': 3, 'CPU#MISCELLANEOUS': 1, 'GRAPHICS#MISCELLANEOUS': 1, 'MULTIMEDIA_DEVICES#USABILITY': 1, 'HARD_DISC#OPERATION_PERFORMANCE': 3, 'SOFTWARE#OPERATION_PERFORMANCE': 2, 'OS#QUALITY': 2, 'PORTS#OPERATION_PERFORMANCE': 1, 'SOFTWARE#DESIGN_FEATURES': 1, 'SOFTWARE#QUALITY': 2, 'WARRANTY#GENERAL': 1, 'BATTERY#DESIGN_FEATURES': 1, 'CPU#GENERAL': 1, 'MOUSE#QUALITY': 2, 'HARD_DISC#QUALITY': 1, 'SOFTWARE#PRICE': 1}
# {'LAPTOP#GENERAL': '0', 'LAPTOP#OPERATION_PERFORMANCE': '1', 'HARD_DISC#DESIGN_FEATURES': '2', 'LAPTOP#QUALITY': '3', 'DISPLAY#QUALITY': '4', 'LAPTOP#DESIGN_FEATURES': '5', 'KEYBOARD#GENERAL': '6', 'LAPTOP#USABILITY': '7', 'LAPTOP#PRICE': '8', 'COMPANY#GENERAL': '9', 'SUPPORT#QUALITY': '10', 'LAPTOP#CONNECTIVITY': '11', 'DISPLAY#OPERATION_PERFORMANCE': '12', 'LAPTOP#PORTABILITY': '13', 'BATTERY#OPERATION_PERFORMANCE': '14', 'MULTIMEDIA_DEVICES#GENERAL': '15', 'MULTIMEDIA_DEVICES#QUALITY': '16', 'MULTIMEDIA_DEVICES#DESIGN_FEATURES': '17', 'MULTIMEDIA_DEVICES#USABILITY': '18', 'KEYBOARD#DESIGN_FEATURES': '19', 'KEYBOARD#USABILITY': '20', 'HARD_DISC#QUALITY': '21', 'WARRANTY#GENERAL': '22', 'MOUSE#DESIGN_FEATURES': '23', 'DISPLAY#GENERAL': '24', 'LAPTOP#MISCELLANEOUS': '25', 'SOFTWARE#GENERAL': '26', 'OS#GENERAL': '27', 'GRAPHICS#QUALITY': '28', 'SHIPPING#QUALITY': '29', 'DISPLAY#DESIGN_FEATURES': '30', 'OS#OPERATION_PERFORMANCE': '31', 'DISPLAY#USABILITY': '32', 'BATTERY#QUALITY': '33', 'POWER_SUPPLY#QUALITY': '34', 'POWER_SUPPLY#OPERATION_PERFORMANCE': '35', 'HARDWARE#QUALITY': '36', 'OS#MISCELLANEOUS': '37', 'GRAPHICS#GENERAL': '38', 'MOUSE#OPERATION_PERFORMANCE': '39', 'WARRANTY#PRICE': '40', 'SHIPPING#PRICE': '41', 'MOUSE#QUALITY': '42', 'POWER_SUPPLY#DESIGN_FEATURES': '43', 'OS#USABILITY': '44', 'KEYBOARD#QUALITY': '45', 'BATTERY#MISCELLANEOUS': '46', 'KEYBOARD#OPERATION_PERFORMANCE': '47', 'MOUSE#USABILITY': '48', 'MEMORY#DESIGN_FEATURES': '49', 'PORTS#DESIGN_FEATURES': '50', 'SOFTWARE#QUALITY': '51', 'OS#QUALITY': '52', 'SOFTWARE#OPERATION_PERFORMANCE': '53', 'SOFTWARE#USABILITY': '54', 'SUPPORT#PRICE': '55', 'MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE': '56', 'MOTHERBOARD#QUALITY': '57', 'CPU#OPERATION_PERFORMANCE': '58', 'MOUSE#GENERAL': '59', 'OPTICAL_DRIVES#QUALITY': '60', 'OS#DESIGN_FEATURES': '61', 'SOFTWARE#MISCELLANEOUS': '62', 'CPU#DESIGN_FEATURES': '63', 'CPU#QUALITY': '64', 'GRAPHICS#DESIGN_FEATURES': '65', 'POWER_SUPPLY#MISCELLANEOUS': '66', 'CPU#MISCELLANEOUS': '67', 'PORTS#QUALITY': '68', 'FANS_COOLING#DESIGN_FEATURES': '69', 'OPTICAL_DRIVES#OPERATION_PERFORMANCE': '70', 'HARDWARE#OPERATION_PERFORMANCE': '71', 'SUPPORT#MISCELLANEOUS': '72', 'FANS_COOLING#QUALITY': '73', 'PORTS#OPERATION_PERFORMANCE': '74', 'GRAPHICS#MISCELLANEOUS': '75', 'FANS_COOLING#OPERATION_PERFORMANCE': '76', 'HARDWARE#GENERAL': '77', 'SOFTWARE#DESIGN_FEATURES': '78', 'SOFTWARE#PRICE': '79', 'MULTIMEDIA_DEVICES#MISCELLANEOUS': '80', 'POWER_SUPPLY#GENERAL': '81', 'OPTICAL_DRIVES#DESIGN_FEATURES': '82', 'HARD_DISC#GENERAL': '83', 'OPTICAL_DRIVES#GENERAL': '84', 'HARD_DISC#OPERATION_PERFORMANCE': '85', 'BATTERY#DESIGN_FEATURES': '86', 'CPU#GENERAL': '87'}