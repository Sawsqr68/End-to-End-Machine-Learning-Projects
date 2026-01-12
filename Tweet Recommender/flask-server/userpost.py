import pandas as pd
from datetime import datetime, timedelta
from twitter_utils import utility


def get_user_history(username, upper_limit=180):
    query = f'from:{username} include:nativeretweets lang:en since:{(datetime.now() - timedelta(days=upper_limit)).strftime("%Y-%m-%d")}'
    return utility(query)

def getuserpost(user_name):
    user_df = get_user_history(user_name)

    return {'data': user_df[['tweet_date', 'content', 'tweet_id', 'username', 'displayname']].to_dict(orient='records')}