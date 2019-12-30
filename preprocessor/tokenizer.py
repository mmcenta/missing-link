import re

from .cleaner import Cleaner


class FullTokenizer:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.URL = re.compile('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+')
        self.cleaner = Cleaner()
        self.simple_tokenizer = SimpleTokenizer()

    def tokenize(self, text):
        urls, contexts = [], []
        text = self.cleaner.clean(text)
        tokens = self.simple_tokenizer.tokenize(text)
        for i, token in enumerate(tokens):
            if self.URL.match(token):
                urls.append(token)
                contexts.append(''.join(tokens[max(0, i - self.window_size):min(len(tokens), i + self.window_size)]))
        return tokens, urls, contexts


class SimpleTokenizer:
    def __init__(self):
        self.WORD = re.compile(r'\w+')

    def tokenize(self, text):
        return self.WORD.findall(text)
