class Source:

    def __init__(self):

        self.name = ''
        self.url = ''

    def set_source(self, name, url):

        self.name = name
        self.url = url

    def get_source(self):

        return {
            "name": self.name,
            "url": self.url
        }
