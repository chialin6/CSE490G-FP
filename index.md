# Politician Tweets Analysis ([Video](https://drive.google.com/file/d/1BSd3sXKjJQrW-jx5LTgutqFANJrx2ROO/view?usp=sharing))
## Introduction

As we all know, Twitter is an important social media among celebrities. Also, recently, it has been a hot topic for text analysis. Since I'm pretty new to natural language processing, I think it would be fun to do some analysis on tweets. With the hottest topic this year except for COVID-19, which is **presidential election**, I decided to crawl few iconic politician candidates from both Republicans and Democrats to see if it is possible to guess a tweet is posted by whom.

![Repulican vs Democrat](https://cdn.cnn.com/cnnnext/dam/assets/181105112842-donkey-elephant-top.jpg)
_This image credits to [CNN](https://www.cnn.com/style/article/why-democrats-are-donkeys-republicans-are-elephants-artsy/index.html)_

Here I crawled five politicians' tweets via Twitter API. These five politicians are Bernie Sanders, Donald Trump, Kamala Harris, Joe Biden, and Mike Pence, respectively. Further details will be described below.


## Method
### Web Scraping
I used the `tweepy` library to scrape the politicians' tweets via Twitter API. For further details please refer to my [jupyter notebook](https://github.com/chialin6/CSE490G-FP/blob/gh-pages/cse490g_fp_details.ipynb). (For security issues, I removed my consumer keys.)
I was planning to scrape 2500 tweets for each politician. However, the Twitter API seems not to work well for `@realDonaldTrump`. Each time, the returned amount varies. At first, I thought that was because of the limitation of API calls. Nonetheless, the API works totally fine for all the other politicians and I also found out someone encountered the same problem as me. ([tweepy Github issue](https://github.com/tweepy/tweepy/issues/1361))
After several times of try and error, I can only get 2193 tweets from Trump to make the dataset as homogeneous as possible.

### Text preprocessing
#### Removal of URLs
Tweets contain a lot of URLs, which happen in both original posts and retweets. However, URLs cannot really represent any semantic meaning, so it is commons to remove URLs while doing tweets analysis. Instead of directly substituting it with spaces, I decided to make it as "URL" to preserve the user behaviors on tweeting links.

#### Removal of Emojis/Emoticons
Emojis is also a new aspect that only applies to tweets. For pure text analysis like news, there's no emojis involved. In fact, emojis are helpful to analyze tweets, according to lots of research over tweets analysis. However, due to technical difficulties, I here decided to remove them and switch them into "emoji".

#### Removal of Punctuations
As for punctuation symbols like `!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~`, these are basic noise when the model trying to understand semantics meaning. As a result, I removed them as well.

#### Lemmatization
Lemmatization is a more special technique compared to text cleansing like above. It combines several words into one word which has similar stemming (root words), for example, "working", "worked", "works" will be assigned as "work". And here I adopted a library called **WordNet** to achieve lemmatization.

### Training
All the further implementation details can be found in my [jupyter notebook](https://github.com/chialin6/CSE490G-FP/blob/gh-pages/cse490g_fp_details.ipynb)
#### Model
![LSTM](https://www.researchgate.net/profile/Savvas_Varsamopoulos/publication/329362532/figure/fig5/AS:699592479870977@1543807253596/Structure-of-the-LSTM-cell-and-equations-that-describe-the-gates-of-an-LSTM-cell.jpg)
_Image credits to [ResearchGate](https://www.researchgate.net/figure/Structure-of-the-LSTM-cell-and-equations-that-describe-the-gates-of-an-LSTM-cell_fig5_329362532)_
In Recurrent Neural Networks, activation outputs are propagated in both directions (from inputs to outputs and from outputs to inputs), which acts as a **memory state** of the neurons. This state allows the neurons an ability to remember what have been learned so far. In **LSTM**, it adopts a new **cell state** which controlled by a forget gate to decide how much it should remember. And thus, LSTM is a really helpful model to process natural language where each word has relationship between its context to some degree.
```python
  self.embedding = nn.Embedding(len(text_field.vocab), emb_dim)
  self.dimension = dimension
  self.lstm = nn.LSTM(input_size=emb_dim,
                      hidden_size=dimension,
                      num_layers=2,
                      batch_first=True,
                      bidirectional=True)
  self.drop = nn.Dropout(p=0.3)

  self.fc = nn.Linear(2*dimension, 5)
```
I deployed a simple LSTM which embeds text into vocab first, and then followed by 2 stacked LSTM cells. I also tried to stack different amount of layers together. However, the best results are similar from one layer to 5 layers.

#### Optimizer
I chose Adam with learning rate = 0.0001 as optimizer.

#### Loss function
For loss function, I used [BCE Loss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) which is good for one-hot encoded multi-class problems.

## Results
The model ends up converging really fast in less than 10 epochs. I've adjusted neural network parameters several times, but they always converge in about 3-5 epochs with **Train Loss: 0.1241, Valid Loss: 0.2091**.
If we converted the predicted array into predicted politicians, it only gets around 20% accuracy, which is basically the same as a random guess.
### Some prediction examples
```
trump predicts 35 growth in q3  ‘we‘re coming back stronger‘ url via breitbartnews would be biggest ever
pred: JoeBiden
label: realDonaldTrump

i urge president trump to join me in saying that while peaceful protest is an important right — violence is wrong period
if he can’t say that then he is unfit to be president url
pred: realDonaldTrump
label: JoeBiden

the launch of the resilience spacecraft is the culmination of four years of renewed leadership in space it’s a tribute to the courage of our astronauts the ingenuity of the men and women of nasa and spacex and the unrelenting vision of president realdonaldtrump
pred: realDonaldTrump
label: Mike_Pence

rt joebiden we have one shot one opportunity one moment
don’t miss the chance — vote url
pred: realDonaldTrump
label: KamalaHarris

if youre excited too join us at 3 pm et url
pred: realDonaldTrump
label: JoeBiden

rt danscavino potus still going in janesville wisconsin maga🦅
url url
pred: realDonaldTrump
label: realDonaldTrump

in the midst of a pandemic which has now claimed 100000 americans’ lives were seeing the reality of trump’s socalled “strong economy” we cannot lose sight of our goal of an economic system that works for all people not just the billionaire class url
pred: realDonaldTrump
label: BernieSanders

from voting rights to health disparities this pandemic is an opportunity to restructure already broken systems that are now too pronounced to ignore 
url
pred: realDonaldTrump
label: KamalaHarris

i know that history has tried to sanitize the movement and to make it a digestible soundbite that rosa sat and martin marched and john crossed a bridge and suddenly we had equality and civil rights we’re still in the civil rights movement” –ayannapressley url
pred: realDonaldTrump
label: BernieSanders

rt realdonaldtrump thank you michigan
url
pred: realDonaldTrump
label: Mike_Pence

rt danscavino potus still going in janesville wisconsin maga🦅
url url
pred: realDonaldTrump
label: realDonaldTrump

roaring support for president realdonaldtrump in wisconsin  url
pred: Mike_Pence
label: Mike_Pence

rt garepublicans its not too late to get your ticket to see vice president mikepence in augusta this afternoon
url…
pred: Mike_Pence
label: Mike_Pence

ignore the polls 
register to vote url url
pred: JoeBiden
label: JoeBiden

rt gop “the last time you saw a democrat nominee for president in the badger state was back in 2012 … i’m here to tell you get used to s…
pred: Mike_Pence
label: Mike_Pence

i know kamalaharris will work every single day to make life better for all americans and build this nation back better than before url
pred: JoeBiden
label: JoeBiden

no the 2020 election was a total scam we won by a lot and will hopefully turn over the fraudulent result but we must get out and help david and kelly two great people otherwise we are playing right into the hands of some very sick people i will be in georgia on saturday url
pred: realDonaldTrump
label: realDonaldTrump

“we believe these people are thieves the big city machines are corrupt this was a stolen election best pollster in britain wrote this morning that this clearly was a stolen election that it’s impossible to imagine that biden outran obama in some of these states
pred: realDonaldTrump
label: realDonaldTrump

adrian perkins has fought for our country and now he wants to take that fight to the senate to build a better louisiana for all watch his video and chip in a few bucks if you can   
url url
pred: KamalaHarris
label: KamalaHarris

as a neuroscientist arati4congress understands health care is a right for all not a privilege arati decided to take on the incumbent democrat in her district because she knows that this moment requires bold progressive leadership i’m proud to endorse her today url
pred: BernieSanders
label: BernieSanders

rt teamtrump president realdonaldtrump jon ossoff is a radical left wing zealot url
pred: realDonaldTrump
label: realDonaldTrump

our announced economic team shares our commitment to building an economy where everyone has access to a higher minimum wage affordable health care paid family leave and paid sick leave 
an america where opportunity is within reach for everyone—for all the people
pred: KamalaHarris
label: KamalaHarris

rt robertcahaly our trafalgargroup gasen runoff polls also shed some light on georgia voters opinions of whether the presidential e…
pred: realDonaldTrump
label: realDonaldTrump

we have great support on the election hoax url
pred: realDonaldTrump
label: realDonaldTrump

west virginians deserve a voice in congress who will look out for the working class and address their economic pain when kunkel4congress gets to washington that is exactly what she will do url
pred: BernieSanders
label: BernieSanders
```

## Conclusions
To be honest, I feel really frustrated that it seems like a bad neural network that learns nothing. There are several reasons that I guess might cause this result.
### Insufficient Dataset
10000 entries of tweets seem like not enough to train a network. It converges too fast to learn, even there's only one layer.
### Too Noisy Input
Although I've done some text-cleansing, the tweets, in general, are still really noisy compared to normal text. Maybe it will be a good idea to try to convert emojis into more meaningful texts or other information.
### High Similarity Among Dataset
I'm not sure whether this might be the reason. Because all the politicians' tweets are similar, it is hard to tell whose tweet is this even when a human is guessing. When I look into the predicted numerical array, it seems like most of the guessing are around 0.5. Only a few are 0.7. I was thinking maybe defining different loss function would help. The other idea will be to have topic/issuee" labels to see which politicians support which topics, instead of just random guessing tweets.
### Other takeaways
I have to say this is not a pleasant result, but I've done my best from collecting data to building a model. Some people also say that a sufficient or a good dataset will actually be helpful, which I believe is true. Instead of just trying to use a neural network to solve any problems, I now think it will be more rational to think about what kind of topics or what types of the dataset would be suitable for a neural network model.

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
14. [Language Modelling and Text Generation using LSTMs — Deep Learning for NLP](https://medium.com/@shivambansal36/language-modelling-text-generation-using-lstms-deep-learning-for-nlp-ed36b224b275)
