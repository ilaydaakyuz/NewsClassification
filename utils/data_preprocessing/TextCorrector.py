from symspellpy.symspellpy import SymSpell
import os

class TextCorrector:
    def __init__(self):
        # SymSpell nesnesi oluştur
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

        # Sözlük dosyalarının yollarını belirleme
        dictionary_path = os.path.join("data", "frequency_dictionary_en_82_765.txt")
        bigram_path = os.path.join("data", "frequency_bigramdictionary_en_243_342.txt")

        # Sözlük dosyalarını yükleme
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        self.sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    def correct_text(self, text):
        """
        Metindeki yazım hatalarını düzeltir.
        """
        if not isinstance(text, str):
            return text
        suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)
        return suggestions[0].term if suggestions else text
