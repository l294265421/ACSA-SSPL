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
train_xml_file_path = data_path.data_base_dir + 'Chinese_phones_training.xml'
train_category_count = {}
train_example = get_all_example_from_xml(train_xml_file_path, category_count=train_category_count)
file_utils.write_lines(head + train_example, data_path.data_base_dir + '/train.csv')

test_xml_file_path = data_path.data_base_dir + 'CH_PHNS_SB1_TEST_.xml'
test_examples = get_all_example_from_xml(test_xml_file_path)
file_utils.write_lines(head + test_examples, data_path.data_base_dir + '/test_public.csv')
file_utils.write_lines(head + test_examples, data_path.data_base_dir + '/test_public_gold.csv')

category_index = {}
for c in category_count_totle.keys():
    category_index[c] = str(len(category_index))
print(category_index)

q1_test_data = []
q2_test_data = []
q3_test_data = []
q4_test_data = []
q_numbers = [1.8, 4.5, 23.3]
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

# sentence_without_opinion_count: 4997
# {'positive': 758, 'negative': 575, 'neutral': 0}
# {'PHONE#DESIGN_FEATURES': 90, 'DISPLAY#QUALITY': 123, 'PORTS#CONNECTIVITY': 7, 'CPU#OPERATION_PERFORMANCE': 13, 'MULTIMEDIA_DEVICES#DESIGN_FEATURES': 26, 'MULTIMEDIA_DEVICES#QUALITY': 69, 'POWER_SUPPLY#QUALITY': 7, 'SOFTWARE#DESIGN_FEATURES': 36, 'HARDWARE#DESIGN_FEATURES': 23, 'PHONE#QUALITY': 56, 'OS#OPERATION_PERFORMANCE': 66, 'PHONE#OPERATION_PERFORMANCE': 133, 'SOFTWARE#OPERATION_PERFORMANCE': 110, 'OS#DESIGN_FEATURES': 39, 'MULTIMEDIA_DEVICES#CONNECTIVITY': 8, 'BATTERY#QUALITY': 46, 'PHONE#GENERAL': 9, 'MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE': 67, 'DISPLAY#DESIGN_FEATURES': 66, 'SUPPORT#USABILITY': 1, 'WARRANTY#OPERATION_PERFORMANCE': 1, 'BATTERY#DESIGN_FEATURES': 2, 'HARDWARE#QUALITY': 36, 'PHONE#CONNECTIVITY': 33, 'KEYBOARD#DESIGN_FEATURES': 30, 'PHONE#PRICE': 12, 'OS#GENERAL': 2, 'HARDWARE#OPERATION_PERFORMANCE': 24, 'CPU#QUALITY': 6, 'MEMORY#QUALITY': 3, 'SOFTWARE#MISCELLANEOUS': 1, 'POWER_SUPPLY#OPERATION_PERFORMANCE': 3, 'BATTERY#OPERATION_PERFORMANCE': 15, 'PORTS#QUALITY': 1, 'DISPLAY#OPERATION_PERFORMANCE': 31, 'OS#USABILITY': 4, 'KEYBOARD#OPERATION_PERFORMANCE': 19, 'SOFTWARE#QUALITY': 11, 'HARD_DISC#QUALITY': 3, 'MULTIMEDIA_DEVICES#GENERAL': 8, 'KEYBOARD#USABILITY': 2, 'OS#CONNECTIVITY': 2, 'SOFTWARE#CONNECTIVITY': 1, 'HARD_DISC#OPERATION_PERFORMANCE': 1, 'HARDWARE#CONNECTIVITY': 2, 'POWER_SUPPLY#CONNECTIVITY': 1, 'PHONE#MISCELLANEOUS': 5, 'MULTIMEDIA_DEVICES#USABILITY': 20, 'DISPLAY#GENERAL': 3, 'SUPPORT#MISCELLANEOUS': 1, 'BATTERY#USABILITY': 2, 'HARDWARE#GENERAL': 3, 'HARDWARE#USABILITY': 5, 'POWER_SUPPLY#GENERAL': 1, 'PHONE#USABILITY': 8, 'DISPLAY#USABILITY': 1, 'MEMORY#USABILITY': 3, 'SOFTWARE#USABILITY': 6, 'PORTS#DESIGN_FEATURES': 3, 'POWER_SUPPLY#USABILITY': 1, 'KEYBOARD#QUALITY': 3, 'OS#QUALITY': 6, 'PORTS#USABILITY': 2, 'CPU#USABILITY': 2, 'PORTS#OPERATION_PERFORMANCE': 2, 'MEMORY#GENERAL': 1, 'CPU#GENERAL': 1, 'KEYBOARD#GENERAL': 1, 'MEMORY#DESIGN_FEATURES': 1, 'OS#PRICE': 1, 'SOFTWARE#GENERAL': 1, 'SUPPORT#QUALITY': 1}
# sentence_without_opinion_count: 2662
# {'positive': 310, 'negative': 219, 'neutral': 0}
# {'SOFTWARE#USABILITY': 12, 'DISPLAY#QUALITY': 69, 'PHONE#OPERATION_PERFORMANCE': 25, 'DISPLAY#MISCELLANEOUS': 1, 'MULTIMEDIA_DEVICES#QUALITY': 49, 'DISPLAY#DESIGN_FEATURES': 18, 'PHONE#DESIGN_FEATURES': 34, 'OS#OPERATION_PERFORMANCE': 21, 'DISPLAY#GENERAL': 7, 'MULTIMEDIA_DEVICES#DESIGN_FEATURES': 9, 'PHONE#MISCELLANEOUS': 18, 'OS#GENERAL': 2, 'PHONE#PORTABILITY': 9, 'HARDWARE#DESIGN_FEATURES': 13, 'OS#USABILITY': 11, 'SOFTWARE#GENERAL': 9, 'BATTERY#OPERATION_PERFORMANCE': 25, 'MULTIMEDIA_DEVICES#CONNECTIVITY': 9, 'PHONE#QUALITY': 48, 'PHONE#USABILITY': 5, 'SOFTWARE#OPERATION_PERFORMANCE': 23, 'OS#QUALITY': 5, 'HARDWARE#CONNECTIVITY': 1, 'PHONE#PRICE': 11, 'SOFTWARE#DESIGN_FEATURES': 7, 'KEYBOARD#DESIGN_FEATURES': 13, 'SUPPORT#QUALITY': 2, 'CPU#OPERATION_PERFORMANCE': 5, 'HARDWARE#GENERAL': 2, 'PHONE#GENERAL': 5, 'HARDWARE#QUALITY': 9, 'MULTIMEDIA_DEVICES#GENERAL': 4, 'SOFTWARE#QUALITY': 4, 'KEYBOARD#OPERATION_PERFORMANCE': 1, 'POWER_SUPPLY#PRICE': 1, 'POWER_SUPPLY#OPERATION_PERFORMANCE': 4, 'MULTIMEDIA_DEVICES#MISCELLANEOUS': 6, 'MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE': 2, 'HARDWARE#MISCELLANEOUS': 3, 'HARDWARE#USABILITY': 1, 'HARDWARE#PRICE': 1, 'HARDWARE#OPERATION_PERFORMANCE': 4, 'SOFTWARE#CONNECTIVITY': 1, 'SOFTWARE#MISCELLANEOUS': 4, 'MULTIMEDIA_DEVICES#USABILITY': 3, 'OS#MISCELLANEOUS': 2, 'BATTERY#QUALITY': 1, 'DISPLAY#USABILITY': 1, 'PHONE#CONNECTIVITY': 1, 'OS#DESIGN_FEATURES': 2, 'BATTERY#GENERAL': 1, 'DISPLAY#OPERATION_PERFORMANCE': 2, 'PORTS#CONNECTIVITY': 1, 'BATTERY#USABILITY': 1, 'DISPLAY#PORTABILITY': 1}
# {'PHONE#DESIGN_FEATURES': '0', 'DISPLAY#QUALITY': '1', 'PORTS#CONNECTIVITY': '2', 'CPU#OPERATION_PERFORMANCE': '3', 'MULTIMEDIA_DEVICES#DESIGN_FEATURES': '4', 'MULTIMEDIA_DEVICES#QUALITY': '5', 'POWER_SUPPLY#QUALITY': '6', 'SOFTWARE#DESIGN_FEATURES': '7', 'HARDWARE#DESIGN_FEATURES': '8', 'PHONE#QUALITY': '9', 'OS#OPERATION_PERFORMANCE': '10', 'PHONE#OPERATION_PERFORMANCE': '11', 'SOFTWARE#OPERATION_PERFORMANCE': '12', 'OS#DESIGN_FEATURES': '13', 'MULTIMEDIA_DEVICES#CONNECTIVITY': '14', 'BATTERY#QUALITY': '15', 'PHONE#GENERAL': '16', 'MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE': '17', 'DISPLAY#DESIGN_FEATURES': '18', 'SUPPORT#USABILITY': '19', 'WARRANTY#OPERATION_PERFORMANCE': '20', 'BATTERY#DESIGN_FEATURES': '21', 'HARDWARE#QUALITY': '22', 'PHONE#CONNECTIVITY': '23', 'KEYBOARD#DESIGN_FEATURES': '24', 'PHONE#PRICE': '25', 'OS#GENERAL': '26', 'HARDWARE#OPERATION_PERFORMANCE': '27', 'CPU#QUALITY': '28', 'MEMORY#QUALITY': '29', 'SOFTWARE#MISCELLANEOUS': '30', 'POWER_SUPPLY#OPERATION_PERFORMANCE': '31', 'BATTERY#OPERATION_PERFORMANCE': '32', 'PORTS#QUALITY': '33', 'DISPLAY#OPERATION_PERFORMANCE': '34', 'OS#USABILITY': '35', 'KEYBOARD#OPERATION_PERFORMANCE': '36', 'SOFTWARE#QUALITY': '37', 'HARD_DISC#QUALITY': '38', 'MULTIMEDIA_DEVICES#GENERAL': '39', 'KEYBOARD#USABILITY': '40', 'OS#CONNECTIVITY': '41', 'SOFTWARE#CONNECTIVITY': '42', 'HARD_DISC#OPERATION_PERFORMANCE': '43', 'HARDWARE#CONNECTIVITY': '44', 'POWER_SUPPLY#CONNECTIVITY': '45', 'PHONE#MISCELLANEOUS': '46', 'MULTIMEDIA_DEVICES#USABILITY': '47', 'DISPLAY#GENERAL': '48', 'SUPPORT#MISCELLANEOUS': '49', 'BATTERY#USABILITY': '50', 'HARDWARE#GENERAL': '51', 'HARDWARE#USABILITY': '52', 'POWER_SUPPLY#GENERAL': '53', 'PHONE#USABILITY': '54', 'DISPLAY#USABILITY': '55', 'MEMORY#USABILITY': '56', 'SOFTWARE#USABILITY': '57', 'PORTS#DESIGN_FEATURES': '58', 'POWER_SUPPLY#USABILITY': '59', 'KEYBOARD#QUALITY': '60', 'OS#QUALITY': '61', 'PORTS#USABILITY': '62', 'CPU#USABILITY': '63', 'PORTS#OPERATION_PERFORMANCE': '64', 'MEMORY#GENERAL': '65', 'CPU#GENERAL': '66', 'KEYBOARD#GENERAL': '67', 'MEMORY#DESIGN_FEATURES': '68', 'OS#PRICE': '69', 'SOFTWARE#GENERAL': '70', 'SUPPORT#QUALITY': '71', 'DISPLAY#MISCELLANEOUS': '72', 'PHONE#PORTABILITY': '73', 'POWER_SUPPLY#PRICE': '74', 'MULTIMEDIA_DEVICES#MISCELLANEOUS': '75', 'HARDWARE#MISCELLANEOUS': '76', 'HARDWARE#PRICE': '77', 'OS#MISCELLANEOUS': '78', 'BATTERY#GENERAL': '79', 'DISPLAY#PORTABILITY': '80'}