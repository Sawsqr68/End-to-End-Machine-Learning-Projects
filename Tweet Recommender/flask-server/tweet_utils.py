"""
Shared utility functions for tweet scraping and processing.
"""
import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import datetime, timedelta


def utility(q, limit=100):
    """
    Scrape tweets based on a query string.
    
    Args:
        q: Query string for Twitter search
        limit: Maximum number of tweets to retrieve (default: 100)
        
    Returns:
        DataFrame containing tweet and user information
    """
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
    """
    Get tweets for a specific topic.
    
    Args:
        topic: Topic string or query
        
    Returns:
        DataFrame containing tweets matching the topic
    """
    query = f'{topic} lang:en'
    return utility(query, 100)


def get_user_history(username, upper_limit=180):
    """
    Get user's tweet history.
    
    Args:
        username: Twitter username
        upper_limit: Number of days to look back (default: 180)
        
    Returns:
        DataFrame containing user's tweets
    """
    query = f'from:{username} include:nativeretweets lang:en since:{(datetime.now() - timedelta(days=upper_limit)).strftime("%Y-%m-%d")}'
    return utility(query)
