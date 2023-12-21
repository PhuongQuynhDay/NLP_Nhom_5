#Các thư viện
import pandas               as pd
import numpy                as np
import string
import re
import warnings
warnings.filterwarnings('ignore')
import nltk
nltk.download("stopwords")
from nltk.corpus                      import stopwords
from nltk.tokenize                    import word_tokenize
from nltk.stem                        import WordNetLemmatizer
from sklearn.model_selection          import train_test_split
from sklearn                          import metrics
from tensorflow                       import keras
from keras.models                     import Sequential
from keras.preprocessing              import sequence
from keras.preprocessing.text         import Tokenizer
from keras.layers                     import Dense, Embedding, LSTM
from keras.preprocessing.sequence     import pad_sequences
from PyQt5                            import QtCore, QtGui, QtWidgets

from background                         import Ui_MainWindow

data = pd.read_csv('D:\Womens Clothing E-Commerce Reviews.csv')
df=data[['Review Text','Recommended IND']]
df = df.dropna(subset = ['Review Text','Recommended IND'])
def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
    text = ' '.join(tokens)
    return text
df['Review Text'] = df['Review Text'].apply(lambda x:clean_text(x))
X = df['Review Text']
y = df['Recommended IND']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle=True)
y_train = np.array(y_train, dtype=np.int32)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train.tolist() + X_test.tolist())

word_index = tokenizer.word_index

X_train_seq = tokenizer.texts_to_sequences(X_train.tolist())

X_train_pad = pad_sequences(X_train_seq)

# Pad the training/test sequences
model = Sequential()
model.add(Embedding(len(word_index) + 1, 1))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(X_train_pad, y_train, epochs=10, batch_size=32)




class Model_main(Ui_MainWindow):
    def __init__(self):
        self.setupUi(MainWindow)
        self.pred.clicked.connect(self.btnPred)
        self.end.clicked.connect(self.btnEnd)
    
    def btnPred(self):
        text = self.input.toPlainText()
        text = clean_text(text)
        if text == '' or text == ' ':
            self.widget.setStyleSheet("background-image: url(:/newPrefix/2.png);")
            return
        X[0:1] = text
        b = tokenizer.texts_to_sequences(X[0:1].tolist())
        b = pad_sequences(b)
        pred = model.predict(b)
        a = pred[0]
        if a > 0.5 :
           print(a)
           self.widget.setStyleSheet("background-image: url(:/newPrefix/3.png);")

        else: 
           print(a)
           self.widget.setStyleSheet("background-image: url(:/newPrefix/4.png);")
    
    def btnEnd(self):
        sys.exit(app.exec_())

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Model_main()
    MainWindow.show()
    sys.exit(app.exec_())
