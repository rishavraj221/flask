# content: string,
# sentiment: object {anger: float, disgust: float, fear: float, joy: float, neutral: float, sadness: float, surprise: float}
# author: object {name: string, url: string, reputation_score: int}
# url: string,
# source: string

from data_structure.Sentiment import Sentiment
from data_structure.Author import Author


class Collection:

    def __init__(self, source):

        self.content = ''
        self.sentiment = Sentiment()
        self.author = Author()
        self.url = ''
        self.source = source

    def set_collection(self, content, url, author_name, author_url):

        self.content = content
        self.url = url
        self.author.set_author(name=author_name, url=author_url)

    def set_sentiment(self, sentiments):

        self.sentiment.set_sentiment(sentiments)

    def get_whole_collection(self):

        return {
            "content": self.content,
            "sentiment": self.sentiment.get_sentiment(),
            "author": self.author.get_author(),
            "url": self.url,
            "source": self.source.get_source()
        }

    def get_content(self):

        return self.content
