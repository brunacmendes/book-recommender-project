from __future__ import print_function, division
from builtins import range, input

from zipfile import ZipFile
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Reshape, Dot
from keras.layers import Add, Activation, Lambda, BatchNormalization, Flatten
from keras.layers import Concatenate, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam, SGD
from keras.regularizers import l2

df = pd.read_csv('datasets/amazon_ratings.csv')

#bookcrossing_data_file_url = (
 #   "http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip"
#)
#BX_zipped_file = keras.utils.get_file(
#    "BX-CSV-Dump.zip", bookcrossing_data_file_url, extract=False
#)
#keras_datasets_path = Path(BX_zipped_file).parents[0]
#BX_dir = keras_datasets_path / 'book-crossing'

# Only extract the data the first time the script is run.
#if not BX_dir.exists():
#    with ZipFile(BX_zipped_file, "r") as zip:
        # Extract files
#        print("Extracting all the files now...")
#        zip.extractall(path=BX_dir)
 #       print("Done!")

#ratings_file = BX_dir / 'BX-Book-Ratings.csv'
#r_cols = ['user_id', 'book_id', 'rating']
#df = pd.read_csv(ratings_file, sep=';', names=r_cols, encoding='latin-1', low_memory=False)
#df = df.drop(df.index[0])

#transforma ratings em valor float
df["rating"] = df["rating"].values.astype(np.float32)


#------apenas para o dataset do bookcrossing
# manter apenas as avaliações explicitas - nota 0 signfica que o usuario apenas interagiu com o produto de alguma forma
#df=df.loc[df['rating']>0]

#colocar ratings que sao de 1-10 para serem de 0.5 a 5
#df['rating']=df["rating"].apply(lambda x: x/2)
#-------------------------------------------------

#tira uma amostra menor do dataset
df = df.sample(frac=0.01, random_state=42)


#transforma os ids dos usuarios e livros 
user_ids = df["user_id"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
book_ids = df["book_id"].unique().tolist()
book2book_encoded = {x: i for i, x in enumerate(book_ids)}
book_encoded2book = {i: x for i, x in enumerate(book_ids)}
df["user"] = df["user_id"].map(user2user_encoded)
df["book"] = df["book_id"].map(book2book_encoded)

n_users = len(user2user_encoded)
n_books = len(book_encoded2book)

# min and max ratings will be used to normalize the ratings later
min_rating = min(df["rating"])
max_rating = max(df["rating"])

print(
    "Number of users: {}, Number of books: {}, Min rating: {}, Max rating: {}".format(
        n_users, n_books, min_rating, max_rating
    )
)

g = df.groupby('user_id')['rating'].count()
top_users = g.sort_values(ascending=False)[:15]

g = df.groupby('book_id')['rating'].count()
top_books = g.sort_values(ascending=False)[:15]

top_r = df.join(top_users, rsuffix='_r', how='inner', on='user_id')
top_r = top_r.join(top_books, rsuffix='_r', how='inner', on='book_id')

print(pd.crosstab(top_r.user_id, top_r.book_id, top_r.rating, aggfunc=np.sum))


# split into train and test

x = df[["user", "book"]].values
# Normalize the targets between 0 and 1. Makes it easy to train.
#y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
y = df["rating"].values
# Assuming training on 90% of the data and validating on 10%.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

n_factors = 25

X_train_array = [X_train[:, 0], X_train[:, 1]]
X_test_array = [X_test[:, 0], X_test[:, 1]]

class EmbeddingLayer:
    def __init__(self, n_items, n_factors):
        self.n_items = n_items
        self.n_factors = n_factors
        
    
    def __call__(self, x):
        x = Embedding(self.n_items, self.n_factors, embeddings_initializer='he_normal',
                      embeddings_regularizer=l2(1e-6))(x)
        x = Reshape((self.n_factors,))(x)
        #x = Flatten()(x)
        return x


def RecommenderNet(n_users, n_books, n_factors, min_rating, max_rating):
    user = Input(shape=(1,))
    u = EmbeddingLayer(n_users, n_factors)(user)
    
    book = Input(shape=(1,))
    m = EmbeddingLayer(n_books, n_factors)(book)

    #as outras caracteristicas dos datasets sao adicionadas como camadas de embedding
    
    x = Concatenate()([u, m])
    x = Dropout(0.05)(x)

    x = Dense(10, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(1, kernel_initializer='he_normal')(x)
    x = Activation('sigmoid')(x)
    x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)

    model = Model(inputs=[user, book], outputs=x)
    opt = Adam(lr=0.0001)
    #opt=SGD(lr=0.01, momentum=0.9)
    model.compile(loss='mean_squared_error', optimizer=opt)

    return model



model = RecommenderNet(n_users, n_books, n_factors, min_rating, max_rating)
model.summary()

history = model.fit(x=X_train_array, 
                    y=y_train, 
                    batch_size=128, 
                    epochs=5,
                    verbose=1, 
                    validation_data=(X_test_array, y_test)
                    )



# plot losses
plt.plot(history.history['loss'], label="train loss")
plt.plot(history.history['val_loss'], label="test loss")
plt.legend()
plt.show()