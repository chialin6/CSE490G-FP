# Politician Tweets Analysis
## Introduction

As we all know, that Twitter is a important social media among celebrities. Also, recently, it has been a hot topic for text analysis. Since I'm pretty new to natural language processing, I think it would be fun to do some analysis on tweets. With the hottest topic this year except COVID-19, which is **presendential election**, I decided to crawl few iconic politician candidates from both repulican and democrat to see if it is possible to guess a tweet is posted by whom.

![Repulican vs Democrat](https://cdn.cnn.com/cnnnext/dam/assets/181105112842-donkey-elephant-top.jpg)
_This image credits to [CNN](https://www.cnn.com/style/article/why-democrats-are-donkeys-republicans-are-elephants-artsy/index.html)_

Here I crawled five politicians' tweets via Twitter API. These five politicians are Bernie Sanders, Donald Trump, Kamala Harris, Joe Biden, and Mike Pence, respectively. Further details will be described below.


## Method
### Web Scraping
I used `tweepy` library to scrape the politicians tweets via Twitter API. Further details please refer to my jupyter file. (For security issue, I removed my consumer keys.)
I was planning to scrape 2500 tweets for each politician. However, the Twitter API seems not working well for `@realDonaldTrump`. Each time, the returned amount varies. At first, I thought that was because of the limitation of API calls. Nonetheless, the API works totally fine for all the other politicians and I also found out someone encountered the same probelm as me. ([tweepy github issue](https://github.com/tweepy/tweepy/issues/1361))
After several times of try and error, I can only get 2193 tweets from Trump to make the dataset as homogeneous as possible.

### Text preprocessing
#### Removal of URLs
#### Removal of Emojis
#### Removal of Puctuations
#### Lemmatization

## Results

## Conclusions

## Reference
1. [A Comprehensive Introduction to Torchtext (Practical Torchtext part 1)](https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/)
2. [LANGUAGE TRANSLATION WITH TORCHTEXT](https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html)
3. [SEQUENCE MODELS AND LONG-SHORT TERM MEMORY NETWORKS](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
4. [LSTM Text Classification Using Pytorch](https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0)
5. [Taming LSTMs: Variable-sized mini-batches and why PyTorch is good for your health](https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e)
6. [BERT Text Classification Using Pytorch](https://towardsdatascience.com/bert-text-classification-using-pytorch-723dfb8b6b5b)
7. [Pytorch torchtext documentation](https://torchtext.readthedocs.io/en/latest/data.html)
8. [Multi-Class Text Classification with LSTM](https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17)
9. [Stackoverflow](https://stackoverflow.com/)
10. [Transfer Learning in NLP for Tweet Stance Classification](https://towardsdatascience.com/transfer-learning-in-nlp-for-tweet-stance-classification-8ab014da8dde)
11. [Beginners Guide to Text Generation using LSTMs](https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms)
12. [NLP FROM SCRATCH: CLASSIFYING NAMES WITH A CHARACTER-LEVEL RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
13. [Getting started with Text Preprocessing](https://www.kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing#Lemmatization)
