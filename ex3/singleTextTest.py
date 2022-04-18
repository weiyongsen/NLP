import joblib
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import sys
import os

TEXT_DATA_DIR = os.path.join('G:\\trainingdata', '20_newsgroup')
labels_index = {}
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[label_id] =name

model = load_model('mytextcnn_model.h5')
texts = []
fpath = os.path.join('G:\\trainingdata', 'validate.txt')
args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
with open(fpath, **args) as f:
    t = f.read()
    i = t.find('\n\n')  # skip header
    if 0 < i:
        t = t[i:]
    texts.append(t)

MAX_SEQUENCE_LENGTH = 1000
tokenizer = joblib.load('token_result.pkl')
x_pred = tokenizer.texts_to_sequences(texts)
x_pred = pad_sequences(x_pred,
                       maxlen=MAX_SEQUENCE_LENGTH)
result = model.predict(x_pred)
print(result)
print(labels_index[result.argmax()])
