"""
Descripttion: DeepModelIPProtection
version: 1.0
Author: XtHhua
Date: 2024-01-19 00:31:20
LastEditors: XtHhua
LastEditTime: 2024-01-20 17:03:32
"""
from collections import defaultdict

from nltk.corpus import wordnet
import thulac


total_synonyms = {
    "男": "男性",
    "大": "宽敞",
    "新": "崭新的",
    "高": "高大",
    "女": "女性",
    "金": "金色",
    "广": "广袤的",
    "小": "渺小",
    "香": "鲜香",
    "通": "通常",
    "老": "老旧",
    "黄": "黄色",
    "银": "银色",
    "顺": "顺利的",
    "热": "滚热的",
    "多": "丰富",
    "长": "不短",
    "黑": "黑色",
    "青": "青色",
    "富": "有钱",
    "实": "实在",
    "白": "白色",
    "密": "密集的",
}


class ADJAttack:
    def __init__(self, path: str) -> None:
        self.thu1 = thulac.thulac()
        self.path = path
        self.total = self.count_adj()

    def find_adjectives(self, sentence):
        text = self.thu1.cut(sentence, text=True)
        adjectives = []
        for token in text.split(" "):
            try:
                word, tag = token.split("_")
                if tag.startswith("a"):
                    adjectives.append(word)
            except:
                pass
        return adjectives

    def get_synonyms(self, word):
        return total_synonyms.get(word, "")

    def replace_adjectives_with_synonyms(self, query_sentence):
        res_sentence = query_sentence
        adjectives = self.find_adjectives(res_sentence)
        for adj in adjectives:
            synonyms = self.get_synonyms(adj)
            if synonyms:
                res_sentence = res_sentence.replace(adj, synonyms)

        return res_sentence

    def count_adj(self):
        l = []
        total = defaultdict()
        with open(self.path, "r") as file:
            for line in file.readlines():
                l.extend(self.find_adjectives(line[0]))
        for w in l:
            total[w] = total.get(w, 0) + 1
        return dict(sorted(total.items(), key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
    adjattack = ADJAttack("./THUCNews/data/source.txt")
    adjs = adjattack.total
    # print(adjs)
    query_sentence = "我的房间很大"
    new_sentence = adjattack.replace_adjectives_with_synonyms(query_sentence)
    print(new_sentence)
