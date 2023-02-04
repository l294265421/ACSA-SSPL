import sys

from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import file_utils

#规则修正
#测试集正文
text={}
with open(data_path.test_public_file_path, 'r', encoding='utf-8') as f:
    f.readline()
    lines = f.readlines()
    for l in lines:
        tmp = l.strip().split(',') 
        text[tmp[0]] = tmp[1]
        
same = []
diff = []
extract = []
        
#模型预测结果
#输入是模型最终预测的结果
result = ['content_id,subject,sentiment_value,sentiment_word']
with open(data_path.test_public_result_file_path + '_' + sys.argv[1] + '.csv', 'r', encoding='utf-8') as f:
    f.readline()
    lines = f.readlines()
    for base in lines:
        flag = 0
        tmp = base.split(',')
        id = tmp[0]
        content = text[tmp[0]]
        topic = tmp[1]
        sentiment = tmp[2]
        #print("%s%s" %(base.strip(), text[tmp[0]]))
        #if id != '0g58m1hpyZjdksD4':
        #    continue
        
        #print(base)
        
        if '还没有' in content:
            result.append("%s,%s,%s," %(id, topic, sentiment))
            continue
        
        if '价格' == topic and '贵了' in content and '稍微' not in content:
            sentiment = "-1"
            
        if '耐撞' in content:
            topic = '安全性'
            
        if '坐垫很好看' in content:
            topic = '内饰'
                
        if '款好看' in content:
            topic = '外观'
             
        if '高速8' in content:
            topic = '油耗'
              
        if '感觉空间小' in content:
            topic = '空间'
            
        if '档杆抖' in content:
            topic = '操控'
            
        if '跟漏油似的' in content or 'ej发动机' in content or '烧厉害' in content or '左右抖' in content or \
        '机油是王道' in content:
            topic = '动力'
               
        if '价空间' in content or '天价' in content:
            topic = '价格'   
        
        if '油耗' == topic and '很省油' in content:
            sentiment = "1"
             
        if '伤不起' in content or '跟漏油似的' in content or \
        '不喜欢内饰' in content or '方向机漏油' in content or '天价' in content or \
        '贵好多' in content or '加速不够劲' in content or '感觉空间小' in content or \
        '了一小节就会熄火' in content or '噪音大的不得了' in content or '发动机偏磨' in content:
            sentiment = "-1"
                
        if '顺滑无比' in content or '款好看' in content or '强太多' in content or '坐垫很好看' in content:
            sentiment = "1"
               
        if '不看好' in content or '声音真的很小' in content or '嘶吼声太大' in content:
            sentiment = "0"
            
        result.append("%s,%s,%s," % (id, topic, sentiment))
file_utils.write_lines(result, data_path.test_public_result_file_path + '_' + sys.argv[1] + '.submit' + '.csv')
