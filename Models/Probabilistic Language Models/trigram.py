import nltk
from nltk import trigrams
from nltk.corpus import reuters
from collections import defaultdict

nltk.download('reuters')
nltk.download('punkt')

words = nltk.word_tokenize(' '.join(reuters.words()))
tri_grams = list(trigrams(words))
model = defaultdict(lambda: defaultdict(lambda: 0))

for w1, w2, w3 in tri_grams:
    model[(w1, w2)][w3] += 1

for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count


def predict_next_word(w1, w2):
    next_word = model[w1, w2]
    if next_word:
        predicted_word = max(next_word, key=next_word.get)
        return predicted_word
    else:
        return "No prediction available"


print("Next Word:", predict_next_word('the', 'stock'))
