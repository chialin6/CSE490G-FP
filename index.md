# Politician Tweets Analysis
## Introduction

As we all know, that Twitter is a important social media among celebrities. Also, recently, it has been a hot topic for text analysis. Since I'm pretty new to natural language processing, I think it would be fun to do some analysis on tweets. With the hottest topic this year except COVID-19, which is **presendential election**, I decided to crawl few iconic politician candidates from both repulican and democrat to see if it is possible to guess a tweet is posted by whom.

![Image](https://cdn.cnn.com/cnnnext/dam/assets/181105112842-donkey-elephant-top.jpg)
*This image credits to [CNN](https://www.cnn.com/style/article/why-democrats-are-donkeys-republicans-are-elephants-artsy/index.html)*

Here I crawled five politicians' tweets via Twitter API. These five politicians are Bernie Sanders, Donald Trump, Kamala Harris, Joe Biden, and Mike Pence, respectively. Further details will be described below.


## Method
### Web Scraping
I used `tweepy` library to scrape the politicians tweets via Twitter API.
```python
# for security purpose, I removed my personal keys grated by Twitter.
consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

politicians = ['BernieSanders', 'realDonaldTrump', 'Mike_Pence', 'JoeBiden', 'KamalaHarris']
count = 2500
tweets_df = pd.DataFrame(columns=['text', 'politician'])

for name in politicians:
    try:
        # Creation of query method using parameters
        tweets = tweepy.Cursor(api.user_timeline, screen_name='@'+name, tweet_mode="extended").items(count)
        tweets_list = [{'text': tweet.full_text, 'politician': name} for tweet in tweets]
        tweets_df = tweets_df.append(tweets_list, ignore_index = True)

    except BaseException as e:
        print('failed on_status,',str(e))
        time.sleep(3)
```


Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

## Results

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/chialin6/cse490g_fp_politician_tweets/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

## Future Improvement

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
