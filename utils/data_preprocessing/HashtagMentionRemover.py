import re

class HashtagMentionRemover:
    @staticmethod
# Hashtag kaldırma fonksiyonu
    def remove_hashtags(text):
        return re.sub(r'[#@]\w+', '', text).strip()



