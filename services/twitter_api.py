import os
import tweepy
from settings import TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_TOKEN_SECRET

# Authenticate to Twitter


class TwitterAPI:
    def __init__(self):
        auth = tweepy.OAuth1UserHandler(
            TWITTER_API_KEY,
            TWITTER_API_SECRET,
            TWITTER_ACCESS_TOKEN,
            TWITTER_TOKEN_SECRET
        )
        self.api = tweepy.API(auth)

    def search_posts(self, keyword, count=10):
        try:
            tweets = self.api.search_tweets(q=keyword, count=count)
            return [{'text': tweet.text, 'user': tweet.user.screen_name} for tweet in tweets]
        except Exception as e:
            print(f'Error searching posts: {e}')
            return []

    def create_post(self, content):
        try:
            tweet = self.api.update_status(content)
            return {'text': tweet.text, 'user': tweet.user.screen_name}
        except Exception as e:
            print(f'Error creating post: {e}')
            return None

    def delete_post(self, tweet_id):
        try:
            self.api.destroy_status(tweet_id)
            return True
        except Exception as e:
            print(f'Error deleting post: {e}')
            return False

    def get_user_timeline(self, user_id, count=10):
        try:
            tweets = self.api.user_timeline(user_id=user_id, count=count)
            return [{'text': tweet.text, 'user': tweet.user.screen_name} for tweet in tweets]
        except Exception as e:
            print(f'Error fetching user timeline: {e}')
            return []


# Example usage of the TwitterAPI class
if __name__ == '__main__':
    twitter_api = TwitterAPI()

    # Example 1: Search for tweets containing a specific keyword
    keyword = 'OpenAI'
    tweets = twitter_api.search_posts(keyword)
    print(f'Search results for "{keyword}":')
    for tweet in tweets:
        print(f'User: {tweet["user"]}, Tweet: {tweet["text"]}')

    # # Example 2: Create a new tweet
    # content = 'Hello Twitter! #myfirsttweet'
    # new_tweet = twitter_api.create_post(content)
    # if new_tweet:
    #     print(f'Created tweet: User: {new_tweet['user']}, Tweet: {new_tweet['text']}')

    # # Example 3: Get user timeline
    # user_id = 'TwitterUserID'
    # user_tweets = twitter_api.get_user_timeline(user_id)
    # print(f'Tweets from user {user_id}:')
    # for tweet in user_tweets:
    #     print(f'User: {tweet['user']}, Tweet: {tweet['text']}')

    # # Example 4: Delete a tweet
    # tweet_id = 'TweetID'
    # if twitter_api.delete_post(tweet_id):
    #     print(f'Tweet with ID {tweet_id} deleted successfully.')
