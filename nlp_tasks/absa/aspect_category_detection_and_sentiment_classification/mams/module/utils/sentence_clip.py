from nlp_tasks.absa.aspect_category_detection_and_sentiment_classification.mams.module.utils.constants import PAD_INDEX

def sentence_clip(sentence):
    mask = (sentence != PAD_INDEX)
    sentence_lens = mask.long().sum(dim=1, keepdim=False)
    max_len = sentence_lens.max().item()
    return sentence[:, :max_len]