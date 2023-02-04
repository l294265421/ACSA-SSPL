# The code and data for the paper "[A Joint Model for Aspect-Category Sentiment Analysis with Shared Sentiment Prediction Layer](https://arxiv.org/abs/1908.11017)"

# Requirements
- Python 3.6.8
- torch==1.3.0
- pytorch-transformers==1.1.0
- allennlp==0.9.0

# Instructions
Before excuting the following commands, replace glove.840B.300d.txt (http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) with the corresponding absolute path of embedding files in your computer. For Chinese datasets, Chinese embedding file is needed and can be downloaded [here](https://github.com/Embedding/Chinese-Word-Vectors).

## Supported Models
- CAE: ACSA-SSL (Aspect-Category Sentiment Analysis Model With Shared Sentiment Prediction Layer)
- CAE-without-cae: ACSA w/o SSL (Aspect-Category Sentiment Analysis Model Without Shared Sentiment Prediction Layer)
- End-to-end cnn: 2018-emnlp-Joint Aspect and Polarity Classification for Adpect-based Sentiment Analysis with End-to-End Neural Networks [paper](papers/2018-Joint%20Aspect%20and%20Polarity%20Classification%20for%20Adpect-based%20Sentiment%20Analysis%20with%20End-to-End%20Neural%20Networks.pdf)
- End-to-end lstm: 2018-emnlp-Joint Aspect and Polarity Classification for Adpect-based Sentiment Analysis with End-to-End Neural Networks [paper](papers/2018-Joint%20Aspect%20and%20Polarity%20Classification%20for%20Adpect-based%20Sentiment%20Analysis%20with%20End-to-End%20Neural%20Networks.pdf)
- AS-Capsules: 2019-www-Aspect-level Sentiment Analysis using AS-Capsules [paper](papers/2019-www-Aspect-level%20Sentiment%20Analysis%20using%20AS-Capsules.pdf)

## Supported Datasets
- SemEval-2016-Task-5-CH-CAME-SB1
- SemEval-2016-Task-5-CH-PHNS-SB1
- SemEval-2016-Task-5-LAPT-SB2
- SemEval-2016-Task-5-REST-SB2

## Run Models
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_cae.py --model_name CAE --current_dataset SemEval-2016-Task-5-REST-SB2 --embedding_filepath glove.840B.300d.txt

# Citation
```
@inproceedings{li-etal-2020-joint-model,
    title = "A Joint Model for Aspect-Category Sentiment Analysis with Shared Sentiment Prediction Layer",
    author = "Li, Yuncong  and
      Yang, Zhe  and
      Yin, Cunxiang  and
      Pan, Xu  and
      Cui, Lunan  and
      Huang, Qiang  and
      Wei, Ting",
    booktitle = "Proceedings of the 19th Chinese National Conference on Computational Linguistics",
    month = oct,
    year = "2020",
    address = "Haikou, China",
    publisher = "Chinese Information Processing Society of China",
    url = "https://aclanthology.org/2020.ccl-1.103",
    pages = "1112--1121",
    abstract = "Aspect-category sentiment analysis (ACSA) aims to predict the aspect categories mentioned in texts and their corresponding sentiment polarities. Some joint models have been proposed to address this task. Given a text, these joint models detect all the aspect categories mentioned in the text and predict the sentiment polarities toward them at once. Although these joint models obtain promising performances, they train separate parameters for each aspect category and therefore suffer from data deficiency of some aspect categories. To solve this problem, we propose a novel joint model which contains a shared sentiment prediction layer. The shared sentiment prediction layer transfers sentiment knowledge between aspect categories and alleviates the problem caused by data deficiency. Experiments conducted on SemEval-2016 Datasets demonstrate the effectiveness of our model.",
    language = "English",
}
```