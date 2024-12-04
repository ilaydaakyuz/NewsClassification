from utils.data_preprocessing.EmojiRemover import EmojiRemover

def main():
    # EmojiRemover class'ını kullan
    emoji_remover = EmojiRemover()
    emoji_remover.clean_text()

if __name__ == "__main__":
    main()

