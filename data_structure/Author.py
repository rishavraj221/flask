class Author:

    def __init__(self):
        self.name = ''
        self.url = ''
        self.reputation_score = 0

    def set_author(self, name, url, reputation_score=0):
        self.name = name
        self.url = url
        self.reputation_score = reputation_score

    def get_author(self):
        return {
            "name": self.name,
            "url": self.url,
            "reputation_score": self.reputation_score
        }
