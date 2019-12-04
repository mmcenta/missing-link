from string import punctuation, digits

class Cleaner:
    def __init__(self, filter=(punctuation + digits)):
        self.filter = filter

    def clean(self, text):
        text = text.replace("\n", " ").replace("\r", " ")
        text = text.translate(str.maketrans(dict.fromkeys(self.filter, " ")))
        return text.translate(str.maketrans(dict.fromkeys("'`", "")))
