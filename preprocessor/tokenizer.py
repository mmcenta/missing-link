import re

from .cleaner import Cleaner


class FullTokenizer:
    def __init__(self):
        self.URL = re.compile('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+')
        self.cleaner = Cleaner()
        self.simple_tokenizer = SimpleTokenizer()

    def tokenize(self, text):
        urls = self.URL.findall(text)
        text = self.cleaner.clean(text)
        tokens = self.simple_tokenizer.tokenize(text)
        return tokens, urls


class SimpleTokenizer:
    def __init__(self):
        self.WORD = re.compile(r'\w+')

    def tokenize(self, text):
        return self.WORD.findall(text)
