from fastapi import FastAPI
from utils import sentiment
from keras.models import load_model
import pickle

model = load_model('models/model.h5')
with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

app = FastAPI()

@app.get("/{Sentence}")
def read_sentence(Sentence):
    return {sentiment(Sentence, tokenizer, model)}