import tweepy
import matplotlib.pyplot as plt
from  textblob import TextBlob 
import time

# all 4 authentication keys to access twitter API
# to connect as OAth handler or jump serever / revers proxy server
consumer_key = "8AO6OU5ubyi4XO47b1C7Sjdlz"
consumer_sec = "FS1usPrfPolvjLXbwGka5N8TWkOZhUsdxGmmTwuO016koesUSt"

# from proxy server we need to connect
access_token = "1151573806680592384-OUFeUtpsRFZM6jQxl1AG99NEjlY0Kt"
access_token_sec = "KKHmkHkDGVaDof8XK4fKKI52DmNl4vZlaXnx85WRfd4Lr"


# tweepy explore
dir(tweepy)

# connected to jump server of twitter
auth=tweepy.OAuthHandler(consumer_key,consumer_sec)

# now we can connect from jump server to web server of twitter
auth.set_access_token(access_token,access_token_sec)


# now we can connect to API storge server of twitter
api_connect=tweepy.API(auth)

# now you can search any topic on twitter
tweet_data=api_connect.search_tweets('Nairobi',count=1000)


pos=0
neg=0
neu=0

# printing line by line
for tweet in tweet_data:
   #print(tweet.text)
   analysis=TextBlob(tweet.text) # here it will apply NLP\
   print(analysis.sentiment)
   # now checking polarity only
   if analysis.sentiment.polarity > 0:
      print("positive")
      pos=pos+1
   elif analysis.sentiment.polarity == 0 :
      print("Neutral")
      neu=neu+1
   else :
      print("Negative")
      neg=neg+1
      
# ploting graphs
plt.xlabel("tags")
plt.ylabel("polarity")
# plt.bar(['Positive','Negative','Neutral'],[pos,neg,neu])
plt.pie([pos,neg,neu],labels=['Positive','Negative','Neutral'],autopct="%1.1f%%")
plt.show()