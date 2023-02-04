# 假设有5个aspect category，输出为5个2分类和5个多分类，二分类用于判断文本有没有提到对应
# aspect category，多分类用于判断对应aspect category的情感
# content_id	content	subject	sentiment_value

ac_aoa = 0
# 2018-Joint Aspect and Polarity Classification for Aspect-based Sentiment Analysis with End-to-End Neural Networks
# 假设有5个aspect category和4个情感类别(正负中冲突)，输出为5个5分类，5分类用于识别文本对于
# 对应aspect category的情感是正负中冲突，还是没有提及
# 2799	and you ca n't beat the prices .	0 0 1 0 0	0 0 0 0 1	0 0 0 0 1	0 0 1 0 0	0 0 0 0 1	0 0 0 0 1
end_to_end_lstm = 1
