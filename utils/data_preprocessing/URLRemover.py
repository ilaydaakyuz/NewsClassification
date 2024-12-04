import re

class URLRemover:
    @staticmethod
    def remove_urls(text):
        # URL'leri düzenli ifadelerle kaldırma
        return re.sub(r'http[s]?://\S+', '', text)
