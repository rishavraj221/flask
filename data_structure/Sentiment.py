class Sentiment:

    def __init__(self):
        self.anger = 0
        self.disgust = 0
        self.fear = 0
        self.joy = 0
        self.neutral = 0
        self.sadness = 0
        self.surprise = 0

    def set_sentiment(self, sentiments):
        self.anger = sentiments.get('anger', 0)
        self.disgust = sentiments.get('disgust', 0)
        self.fear = sentiments.get('fear', 0)
        self.joy = sentiments.get('joy', 0)
        self.neutral = sentiments.get('neutral', 0)
        self.sadness = sentiments.get('sadness', 0)
        self.surprise = sentiments.get('surprise', 0)

    def get_sentiment(self):
        return {
            "anger": self.anger,
            "disgust": self.disgust,
            "fear": self.fear,
            "joy": self.joy,
            "neutral": self.neutral,
            "sadness": self.sadness,
            "surprise": self.surprise
        }
