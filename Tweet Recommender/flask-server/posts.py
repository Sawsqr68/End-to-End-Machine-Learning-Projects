
import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer


import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# Cache stopwords to avoid repeated file I/O
_cached_stopwords = None

def get_stopwords():
    """Load and cache stopwords."""
    global _cached_stopwords
    if _cached_stopwords is None:
        stop_words = []
        try:
            with open("StopWords_Generic.txt", "r") as f:
                stop_text = f.read()
                stop_words_upper = stop_text.split("\n")
                for word in stop_words_upper:
                    stop_words.append(word.lower())
        except FileNotFoundError:
            pass  # File may not exist, continue with NLTK stopwords only
        
        for word in stopwords.words('english'):
            if word not in stop_words:
                stop_words.append(word)
        
        _cached_stopwords = stop_words
    return _cached_stopwords


def utility(q, limit=100):
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(q).get_items():

        if len(tweets) == limit:
            break

        else:
            #tweet details
            try:
                url = tweet.url
            except AttributeError:
                print('AttributeError @url')
                url = None
            
            try:
                tweetDate = tweet.date
            except AttributeError:
                print('AttributeError @tweetDate')
                tweetDate = None 

            try:
                content = tweet.content
            except AttributeError:
                print('AttributeError @content')
                content = None 

            try:
                tweetId = str(tweet.id)
            except AttributeError:
                print('AttributeError @id')
                tweetId = None 

            try:
                replyCount = int(tweet.replyCount)
            except AttributeError:
                print('AttributeError @replyCount')
                replyCount = 0 

            try:
                if tweet.retweetCount:
                    retweetCount = int(tweet.retweetCount)
                else:
                    retweetCount = 0
            except AttributeError:
                print('AttributeError @retweetCount')
                retweetCount = 0 

            try:
                likeCount = int(tweet.likeCount)
            except AttributeError:
                print('AttributeError @likeCount')
                likeCount = 0 

            try:
                quoteCount = int(tweet.quoteCount)
            except AttributeError:
                print('AttributeError @quoteCount')
                quoteCount = None 

            try:
                conversationId = str(tweet.conversationId)
            except AttributeError:
                print('AttributeError @conversationId')
                conversationId = None 

            try:
                language = tweet.lang
            except AttributeError:
                print('AttributeError @language')
                language = None 

            try:
                outlinks = tweet.outlinks
            except AttributeError:
                print('AttributeError @outlinks')
                outlinks = None 

            try:
                media = tweet.media
            except AttributeError:
                print('AttributeError @media')
                media = None 

            try:
                retweetedTweet = str(tweet.retweetedTweet)
            except AttributeError:
                print('AttributeError @retweetedTweet')
                retweetedTweet = None 
            
            try:
                quotedTweet = str(tweet.quotedTweet)
            except AttributeError:
                print('AttributeError @quotedTweet')
                quotedTweet = None 

            try: 
                inReplyToTweetId = str(tweet.inReplyToTweetId)
            except AttributeError:
                print('AttributeError @inReplyToTweetId')
                inReplyToTweetId = None 

            try:
                mentionedUsers = tweet.mentionedUsers
            except AttributeError:
                print('AttributeError @mentionedUsers')
                mentionedUsers = None 

            try:
                hashtags = tweet.hashtags
            except AttributeError:
                print('AttributeError @hashtags')
                hashtags = None 

            try:
                place = tweet.place
            except AttributeError:
                print('AttributeError @place')
                place = None 

            #user details

            try:
                userId = str(tweet.user.id)
            except AttributeError:
                print('AttributeError @userid')
                userId = None 

            try: 
                username = tweet.user.username
            except AttributeError:
                print('AttributeError @username')
                username = None 

            try:
                displayname = tweet.user.displayname
            except AttributeError:
                print('AttributeError @displayname')
                displayname = None 

            try:
                description = tweet.user.description
            except AttributeError:
                print('AttributeError @userDescription')
                description = None 

            try:
                descriptionUrls = tweet.user.descriptionUrls
            except AttributeError:
                print('AttributeError @descriptionUrls')
                descriptionUrls = None 

            try: 
                profileCreated = tweet.user.created
            except AttributeError:
                print('AttributeError @profileCreated')
                profileCreated = None 

            try: 
                followersCount = int(tweet.user.followersCount)
            except AttributeError:
                print('AttributeError @followersCount')
                followersCount = None 

            try:
                friendsCount = int(tweet.user.friendsCount)
            except AttributeError:
                print('AttributeError @friendsCount')
                friendsCount = None 

            try:
                statusesCount = int(tweet.user.statusesCount)
            except AttributeError:
                print('AttributeError @statusesCount')
                statusesCount = None 

            try:
                favouritesCount = int(tweet.user.favouritesCount)
            except AttributeError:
                print('AttributeError @favoritesCount')
                favouritesCount = None 

            try:
                listedCount = int(tweet.user.listedCount)
            except AttributeError:
                print('AttributeError @listedCount')
                listedCount = None 

            try:
                mediaCount = int(tweet.user.mediaCount)
            except AttributeError:
                print('AttributeError @mediaCount')
                mediaCount = None 

            try:
                location = tweet.user.location
            except AttributeError:
                print('AttributeError @location')
                location = None 
            
            try:  
                linkUrl = tweet.user.linkUrl
            except AttributeError:
                print('AttributeError @linkUrl')
                linkUrl = None 


            tweets.append([url,
                        tweetDate, # unix epoch timestamp in milliseconds
                        content,
                        tweetId,
                        replyCount,
                        retweetCount,
                        likeCount,
                        quoteCount,
                        conversationId,
                        language,
                        retweetedTweet,
                        quotedTweet,
                        inReplyToTweetId,
                        userId,
                        username,
                        displayname,
                        description,
                        followersCount,
                        friendsCount,
                        statusesCount,
                        favouritesCount,
                        listedCount,
                        mediaCount])

    tweets_df = pd.DataFrame(tweets, columns=['url',
                        'tweet_date',
                        'content',
                        'tweet_id',
                        'reply_count',
                        'retweet_count',
                        'like_count',
                        'quote_count',
                        'conversation_id',
                        'language',
                        'retweeted_tweet_id',
                        'quoted_tweet_id',
                        'inreply_to_tweet_id',
                        'user_id',
                        'username',
                        'displayname',
                        'description',
                        'followers_count',
                        'friends_count',
                        'statuses_count',
                        'favourites_count',
                        'listed_count',
                        'media_count'])


    return tweets_df

def get_topic_specific_tweets(topic):
    query = f'{topic} lang:en'
    return utility(query, 100)

def get_user_history(username, upper_limit=180):
    query = f'from:{username} include:nativeretweets lang:en since:{(datetime.now() - timedelta(days=upper_limit)).strftime("%Y-%m-%d")}'
    return utility(query)


def get_posts(user_name, topics):
    final_topics = "("
    for val in topics.split(","):
        final_topics += f'"{val.strip()}" OR '
    final_topics = final_topics[:-4]
    final_topics += ")"
    print(final_topics)
    topic_tweets = get_topic_specific_tweets(final_topics)
    user_df = get_user_history(user_name)

    merged_df = pd.concat([topic_tweets], axis=0)
    merged_df = merged_df.reset_index(drop=True)
    merged_df = merged_df.rename(columns={'content': 'tweet_content'})

    stop_words = get_stopwords()

    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(merged_df['tweet_content'].tolist())

    vectorized_df = pd.DataFrame(X.toarray(), columns=list(vectorizer.get_feature_names_out()))

    final_df = pd.concat([merged_df, vectorized_df], axis=1)
    del vectorized_df
    del merged_df

    # Vectorize user content once
    user_content = user_df['content'].tolist()
    user_vectors = vectorizer.transform(user_content).toarray()
    
    # Use vectorized operations instead of nested loops
    tweet_vectors = final_df.iloc[:, -user_vectors.shape[1]:].values
    
    # Compute cosine similarities using matrix operations
    # This replaces the O(n*m) nested loop with vectorized operations
    similarities = np.zeros(len(final_df))
    for user_vec in user_vectors:
        # Compute similarity with all tweets at once
        for j in range(len(final_df)):
            similarities[j] += (1 - spatial.distance.cosine(user_vec, tweet_vectors[j]))
    
    # Normalize by number of user tweets
    scores = similarities / len(user_df)

    final_df['tweet_scores'] = pd.Series(scores)
    final_df = final_df.sort_values('tweet_scores', ascending=False)
    final_df = final_df.reset_index(drop=True)
    final_df = final_df.head(20)

    perc_25 = np.percentile(final_df['tweet_scores'].tolist(), 25)
    perc_75 = np.percentile(final_df['tweet_scores'].tolist(), 75)

    perc_scores = []
    for i in range(len(final_df)):
        if final_df.iloc[i]['tweet_scores'] < perc_25:
            perc_scores.append('A')
        elif final_df.iloc[i]['tweet_scores'] < perc_75:
            perc_scores.append('B')
        else:
            perc_scores.append('C')

    final_df['percentile_scores'] = pd.Series(perc_scores)
    


    # dicty = {}

    # for col in ['tweet_date', 'tweet_content', 'username', 'displayname']:
    #     dicty[col] = final_df.head(20)[col].astype('str').tolist()

    # dicty['user_df_len'] = len(user_df)
    # dicty['topic_tweets_len'] = len(topic_tweets)

    return {'data': final_df[['tweet_date', 'tweet_content', 'tweet_id', 'username', 'displayname', 'percentile_scores']].to_dict(orient='records')}


def getPopularPost(t_id, user_name, user_topic):
    print(t_id, user_name, user_topic)
    """
    input: tweet_id, user name and user query
    output: top 10 most similar + popular tweets
    """
    user_df = get_user_history(user_name)
    
    stweet = user_df[user_df.tweet_id == t_id]

    min_replies = stweet['reply_count'].iloc[0]
    min_faves = stweet['like_count'].iloc[0]
    min_retweets = stweet['retweet_count'].iloc[0]
    print(min_replies, min_faves, min_retweets)

    user_tweet_query = '("' + '" OR "'.join([word.strip() for word in user_topic.split(",")]) + '")' +  f' min_replies:{min_replies} min_faves:{min_faves} min_retweets:{min_retweets} -from:{user_name} lang:en'
    
    # user_tweet_query += f' -from:{user_name} lang:en'
    
    topic_tweets = get_topic_specific_tweets(user_tweet_query)
    
    merged_df = pd.concat([topic_tweets], axis=0)
    merged_df = merged_df.reset_index(drop=True)
    merged_df = merged_df.rename(columns={'content': 'tweet_content'}) 
    print("Got data", merged_df.shape)   
    
    # stop_words = []

    # f = open("StopWords_Generic.txt", "r")
    # stop_text = f.read()
    #stop_words_upper = stop_text.split("\n")
    #for word in stop_words_upper:
    #    stop_words.append(word.lower())

    #for word in stopwords.words('english'):
    #    if word not in stop_words:
    #        stop_words.append(word)
            
    
    # vectorizer = TfidfVectorizer(stop_words=stop_words)
    # X = vectorizer.fit_transform(merged_df['tweet_content'].tolist())

    # vectorized_df = pd.DataFrame(X.toarray(), columns=list(vectorizer.get_feature_names_out()))

    model = SentenceTransformer('bert-base-nli-mean-tokens')
    vectorized_df = pd.DataFrame(model.encode(merged_df['tweet_content'].tolist()))

    final_df = pd.concat([merged_df, vectorized_df], axis=1)
    print("Encoding done")

    sim_scores = [0] * len(final_df)
    
    content = user_df[user_df.tweet_id == t_id]['content']
    
    # pred = vectorizer.transform([content.iloc[0]]).toarray()[0]
    pred = model.encode([content.iloc[0]])[0]
    
    for j in range(len(final_df)):
        row = list(final_df.iloc[j])[-len(pred):]
        sim_scores[j] += (1 - spatial.distance.cosine(pred, row))
    
    final_df['scores'] = pd.Series(sim_scores)
    final_df = final_df.sort_values('scores', ascending=False)
    final_df = final_df.reset_index(drop=True)
    
    print("scores done")
    
    top = 20
    
    most_similar_tweets_df = final_df.iloc[:top,:]
    
    w1 = 0.5
    w2 = 0.3
    w3 = 0.2
    
    most_similar_tweets_df.loc[:,'popularity'] = ((w1*most_similar_tweets_df.loc[:,'reply_count']) + (w2*most_similar_tweets_df.loc[:,'retweet_count']) + (w3*most_similar_tweets_df.loc[:,'like_count'])) / (most_similar_tweets_df.loc[:,'followers_count'])
    most_similar_tweets_df = most_similar_tweets_df.sort_values('popularity', ascending=False)
    most_similar_tweets_df = most_similar_tweets_df.reset_index(drop=True)
    most_similar_tweets_df = most_similar_tweets_df.head(10)
    
    return {'data': most_similar_tweets_df[['tweet_id', 'tweet_date', 'username', 'displayname', 'tweet_content', 'like_count', 'reply_count', 'retweet_count', 'quote_count', 'popularity']].to_dict(orient='records'), 'likes': most_similar_tweets_df['like_count'].mean(), 'replies': most_similar_tweets_df['reply_count'].mean(), 'retweets': most_similar_tweets_df['retweet_count'].mean(), 'min_faves': int(min_faves), 'min_replies': int(min_replies), 'min_retweets': int(min_retweets)}