from dataclasses import dataclass
import pandas as pd

@dataclass
class Verse:
    text: str
    citation: str
    index: int

class ScriptureStorage:
    def __init__(self, file):
        self.df = pd.read_csv(file)

    def get_verse(self, idx: int) -> Verse:
        row = self.df.iloc[idx]

        verse = Verse(text=row['scripture_text'],
                      citation=row['verse_title'],
                      index=idx)
        
        return verse
    
    def get_verses(self, indices: list[int]) -> list[Verse]:
        verses = []
        for idx in indices:
            verse = self.get_verse(idx)
            verses.append(verse)

        return verses
    
    def get_all_texts(self) -> list[str]:
        verses = self.df['scripture_text'].to_list()

        return verses
