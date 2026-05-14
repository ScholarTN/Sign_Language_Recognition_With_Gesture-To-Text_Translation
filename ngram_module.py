import nltk
from nltk.corpus import brown
from collections import Counter
import math

class NgramLM:
    def __init__(self):
        self.unigram = Counter()
        print("Building n-gram model...")
        for sent in brown.sents(categories=["news"]):
            for word in sent:
                self.unigram[word.lower()] += 1
        print("Vocabulary size:", len(self.unigram))

    def complete(self, prefix, top_k=5):
        prefix = prefix.lower()
        words = [w for w in self.unigram if w.startswith(prefix)]
        words.sort(key=lambda w: -self.unigram[w])
        return words[:top_k]