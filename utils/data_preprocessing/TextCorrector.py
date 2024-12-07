from symspellpy.symspellpy import SymSpell
import os

class TextCorrector:
    def __init__(self, project_root=None):
        # SymSpell nesnesi oluştur
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

        # Proje kök dizinini al
        self.project_root = project_root if project_root else os.getcwd()

        # Sözlük dosyalarının yollarını belirleme
        dictionary_path = os.path.join(self.project_root, "data", "frequency_dictionary_en_82_765.txt")
        bigram_path = os.path.join(self.project_root, "data", "frequency_bigramdictionary_en_243_342.txt")

        # Sözlük dosyalarını yükleme
        if not os.path.exists(dictionary_path):
            raise FileNotFoundError(f"Frequency dictionary dosyası bulunamadı: {dictionary_path}")
        if not os.path.exists(bigram_path):
            raise FileNotFoundError(f"Bigram dictionary dosyası bulunamadı: {bigram_path}")

        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        self.sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    def correct_text(self, text):
        """
        Metindeki yazım hatalarını düzeltir.
        """
        if not isinstance(text, str):
            return text  # Eğer metin değilse, olduğu gibi döndür
        suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)
        return suggestions[0].term if suggestions else text
