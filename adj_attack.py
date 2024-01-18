import pkuseg
from nltk.corpus import wordnet


class ADJAttack:
    def __init__(self, query_sentence: str) -> None:
        self.sentence = query_sentence

    def find_adjectives(self, sentence):
        seg = pkuseg.pkuseg()
        words = seg.cut(sentence)

        adjectives = []
        for word, pos in words:
            if pos.startswith("a"):  # 'a' 表示形容词的词性标记
                adjectives.append(word)

        return adjectives

    def get_synonyms(self, word):
        synonyms = set()
        for synset in wordnet.synsets(word, pos=wordnet.ADJ):
            for lemma in synset.lemmas():
                synonyms.add(lemma.name())
        return synonyms

    def replace_adjectives_with_synonyms(self):
        res_sentence = self.sentence
        adjectives = self.find_adjectives(res_sentence)
        for adj in adjectives:
            synonyms = self.get_synonyms(adj)
            if synonyms:
                new_adj = synonyms.pop()  # 选择一个近义词替换
                res_sentence = res_sentence.replace(adj, new_adj)

        return res_sentence


if __name__ == "__main__":
    input_sentence = "这只可爱的小猫在阳光下打盹。"
    adjattack = ADJAttack(input_sentence)
    output_sentence = adjattack.replace_adjectives_with_synonyms()
    print("原始句子:", input_sentence)
    print("替换后的句子:", output_sentence)
