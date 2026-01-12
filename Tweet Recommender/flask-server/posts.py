
import pandas as pd
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from twitter_utils import utility


import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# Cache stopwords to avoid reading file on every request
_cached_stopwords = None

def get_stopwords():
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
            print("Warning: StopWords_Generic.txt not found. Using only NLTK stopwords.")
        except Exception as e:
            print(f"Warning: Error reading StopWords_Generic.txt: {e}. Using only NLTK stopwords.")
        
        for word in stopwords.words('english'):
            if word not in stop_words:
                stop_words.append(word)
        
        _cached_stopwords = stop_words
    return _cached_stopwords


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

    scores = np.zeros(len(final_df))

    # Vectorize the similarity calculation for better performance using numpy operations
    final_df_vectors = final_df.iloc[:, -len(vectorizer.get_feature_names_out()):].values
    
    for i in range(len(user_df)):
        content = user_df.iloc[i]['content']
        pred = vectorizer.transform([content]).toarray()[0]
        # Calculate cosine similarity for all rows at once using numpy dot product
        pred_norm = np.linalg.norm(pred)
        if pred_norm > 0:
            vector_norms = np.linalg.norm(final_df_vectors, axis=1)
            # Avoid division by zero
            valid_norms = vector_norms > 0
            similarities = np.zeros(len(final_df_vectors))
            similarities[valid_norms] = np.dot(final_df_vectors[valid_norms], pred) / (vector_norms[valid_norms] * pred_norm)
            scores += similarities

    scores = scores / len(user_df)
    scores = scores.tolist()

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