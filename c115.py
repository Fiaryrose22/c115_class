import tensorflow.keras
import numpy as np 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequnce import pad_senquences

sentence = ["I am happy to meet my frinds. We are plnning to go a party",
            "I had bd day at school. I got hurt while playing at football"]

tokenizer = Tokenizer(num_words =10000, oov_token='<OOV>')
tokenizer.fit_on_text(sentence)

word_index = tokenizer.word_index

sentence = tokenizer.texts_to_sequnces(sentence)

padded = pad_senquences(sentence,maxlen = 100,
                        padding='post',truncating='post')

print(padded[0:2])

model = tensorflow.keras.models.load_model('Text_Emontion.h5')

result = model.predict(padded)
print(result)

predict_class = np.argmax(result, axis=1)

print(predict_class)