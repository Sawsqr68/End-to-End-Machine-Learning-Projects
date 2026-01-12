"""
Shared utilities for tweet scraping and processing.
This module extracts the common utility function used by both posts.py and userpost.py
to avoid code duplication.
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
        DataFrame containing tweet data
    """
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(q).get_items():

        if len(tweets) == limit:
            break

        else:
            # Use getattr with default values for cleaner code
            url = getattr(tweet, 'url', None)
            tweetDate = getattr(tweet, 'date', None)
            content = getattr(tweet, 'content', None)
            tweetId = str(tweet.id) if hasattr(tweet, 'id') else None
            replyCount = int(tweet.replyCount) if hasattr(tweet, 'replyCount') else 0
            
            # Handle retweetCount with fallback
            if hasattr(tweet, 'retweetCount'):
                retweetCount = int(tweet.retweetCount) if tweet.retweetCount else 0
            else:
                retweetCount = 0
                
            likeCount = int(tweet.likeCount) if hasattr(tweet, 'likeCount') else 0
            quoteCount = int(tweet.quoteCount) if hasattr(tweet, 'quoteCount') else None
            conversationId = str(tweet.conversationId) if hasattr(tweet, 'conversationId') else None
            language = getattr(tweet, 'lang', None)
            retweetedTweet = str(tweet.retweetedTweet) if hasattr(tweet, 'retweetedTweet') else None
            quotedTweet = str(tweet.quotedTweet) if hasattr(tweet, 'quotedTweet') else None
            inReplyToTweetId = str(tweet.inReplyToTweetId) if hasattr(tweet, 'inReplyToTweetId') else None

            # User details - use getattr for safer access
            userId = str(tweet.user.id) if hasattr(tweet, 'user') and hasattr(tweet.user, 'id') else None
            username = tweet.user.username if hasattr(tweet, 'user') and hasattr(tweet.user, 'username') else None
            displayname = tweet.user.displayname if hasattr(tweet, 'user') and hasattr(tweet.user, 'displayname') else None
            description = tweet.user.description if hasattr(tweet, 'user') and hasattr(tweet.user, 'description') else None
            followersCount = int(tweet.user.followersCount) if hasattr(tweet, 'user') and hasattr(tweet.user, 'followersCount') else None
            friendsCount = int(tweet.user.friendsCount) if hasattr(tweet, 'user') and hasattr(tweet.user, 'friendsCount') else None
            statusesCount = int(tweet.user.statusesCount) if hasattr(tweet, 'user') and hasattr(tweet.user, 'statusesCount') else None
            favouritesCount = int(tweet.user.favouritesCount) if hasattr(tweet, 'user') and hasattr(tweet.user, 'favouritesCount') else None
            listedCount = int(tweet.user.listedCount) if hasattr(tweet, 'user') and hasattr(tweet.user, 'listedCount') else None
            mediaCount = int(tweet.user.mediaCount) if hasattr(tweet, 'user') and hasattr(tweet.user, 'mediaCount') else None

            tweets.append([url,
                        tweetDate,
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


def get_user_history(username, upper_limit=180):
    """
    Get tweet history for a specific user.
    
    Args:
        username: Twitter username
        upper_limit: Number of days to look back (default: 180)
        
    Returns:
        DataFrame containing user's tweet history
    """
    query = f'from:{username} include:nativeretweets lang:en since:{(datetime.now() - timedelta(days=upper_limit)).strftime("%Y-%m-%d")}'
    return utility(query)


def get_topic_specific_tweets(topic, limit=100):
    """
    Get tweets for a specific topic.
    
    Args:
        topic: Topic or query string
        limit: Maximum number of tweets (default: 100)
        
    Returns:
        DataFrame containing topic-specific tweets
    """
    query = f'{topic} lang:en'
    return utility(query, limit)
