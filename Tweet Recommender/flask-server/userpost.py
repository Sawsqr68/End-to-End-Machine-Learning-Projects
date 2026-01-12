import pandas as pd
from tweet_utils import get_user_history

def getuserpost(user_name):
    user_df = get_user_history(user_name)

    return {'data': user_df[['tweet_date', 'content', 'tweet_id', 'username', 'displayname']].to_dict(orient='records')}