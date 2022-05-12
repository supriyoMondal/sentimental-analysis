from keras.preprocessing.sequence import pad_sequences
import numpy as np
from constants import max_length, trunc_type, padding_type


def sentiment(Sentence, tokenizer, model):
    Sentence = tokenizer.texts_to_sequences([Sentence])
    Sentence = pad_sequences(Sentence,
                           maxlen = max_length,
                           padding = padding_type,
                           truncating = trunc_type)
    ans = np.argmax(model.predict(Sentence), axis = -1)[0]
    if ans == 0: return 'Very Bad Movie'
    elif ans == 1: return 'Bad Movie'
    elif ans==2: return 'Average Movie'
    elif ans == 3: return 'Good Movie'
    else: return 'Best Movie'