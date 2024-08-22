import nltk
from nltk import bigrams
from nltk.corpus import reuters
from collections import defaultdict

nltk.download('reuters')
nltk.download('punkt')

words = nltk.word_tokenize(' '.join(reuters.words()))
bi_grams = list(bigrams(words))
model = defaultdict(lambda: defaultdict(lambda: 0))

for w1, w2 in bi_grams:
    model[w1][w2] += 1

for w1 in model:
    total_count = float(sum(model[w1].values()))
    for w2 in model[w1]:
        model[w1][w2] /= total_count


def predict_next_word(w1):
    next_word = model[w1]
    if next_word:
        predicted_word = max(next_word, key=next_word.get) 
        return predicted_word
    else:
        return "No prediction available"


print("Next Word:", predict_next_word('the'))
