
import pandas as pd
from keras.layers import * #Input, Embedding, Dense,Flatten, merge,Activation
from keras.models import Model
from keras.regularizers import l2 as l2_reg
import itertools
import keras
from keras.optimizers import *
from keras.regularizers import l2

train_data = pd.read_csv('rating.csv',sep=',',header=0)
train_data = train_data.sample(frac=1.0)
train_data['rating'] = train_data['rating']/5.0
feat_cols = []
cat_cols = []
from sklearn import preprocessing
ule = preprocessing.LabelEncoder()
vle = preprocessing.LabelEncoder()
for feat in train_data.columns:
	if feat in ['userId','movieId']:
		if feat == 'userId':
			le = ule
		else:
			le = vle
		feat_cols.append(feat)
		le.fit(train_data[feat])
		train_data["new_%s"%feat] = le.transform(train_data[feat])
		cat_cols.append("new_%s"%feat)
print(feat_cols)
print(cat_cols)
print(train_data[cat_cols])
print(train_data)
from keras.utils import plot_model
def KerasFM(max_features,K=8,solver=Adam(lr=0.01),l2=0.00,l2_fm = 0.00):
    inputs = []
    flatten_layers=[]
    columns = range(len(max_features))
    fm_layers = []
    #for c in columns:
    for c in max_features.keys():
        inputs_c = Input(shape=(1,), dtype='int32',name = 'input_%s'%c)
        num_c = max_features[c]
        embed_c = Embedding(
                        num_c,
                        K,
                        input_length=1,
                        name = 'embed_%s'%c,
			embeddings_regularizer=keras.regularizers.l2(1e-5)
                        )(inputs_c)

        flatten_c = Flatten()(embed_c)

        inputs.append(inputs_c)
        flatten_layers.append(flatten_c)
    for emb1,emb2 in itertools.combinations(flatten_layers, 2):
        dot_layer = dot([emb1,emb2],axes=-1,normalize=True)
        fm_layers.append(dot_layer)

    #flatten = BatchNormalization(axis=1)(add((fm_layers)))
    flatten = dot_layer
    outputs = Dense(1,activation='linear',name='outputs')(flatten)
    model = Model(input=inputs, output=outputs)
    model.compile(optimizer=solver,loss= 'mae')
    #plot_model(model, to_file='fm_cosine_model.png',show_shapes=True)
    model.summary()
    return model


max_features = train_data[cat_cols].max() + 1

train_len = int(len(train_data)*0.95)
X_train, X_valid = train_data[cat_cols][:train_len], train_data[cat_cols][train_len:]
y_train, y_valid = train_data['rating'][:train_len], train_data['rating'][train_len:]

train_input = []
valid_input = []
#test_input = []
#print(test_data)
for col in cat_cols:
    train_input.append(X_train[col])
    valid_input.append(X_valid[col])
#    test_input.append(test_data[col])
ck = keras.callbacks.ModelCheckpoint("best.model", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
model = KerasFM(max_features)
model.fit(train_input, y_train, batch_size=100000,nb_epoch=2,validation_data=(valid_input,y_valid),callbacks=[ck,es])

from sklearn.metrics import roc_auc_score
p_valid = model.predict(valid_input)
auc = roc_auc_score(y_valid>=0.75,p_valid)
print("valid auc is %0.6f"%auc)
p_valid = p_valid*5
from sklearn.metrics import *
mse = mean_absolute_error(y_valid*5.0,p_valid)
print("valid mae is %0.6f"%mse)
valid_df = train_data[feat_cols][train_len:]
valid_df['rating'] = 5.0*train_data['rating'][train_len:]
valid_df['pred_rating'] = p_valid
valid_df.to_csv('cosine_valid.csv',index=False)

#######################
#start FM u2i
#model.summary()

import faiss
from faiss import normalize_L2
photo_emb_layer = model.get_layer('embed_new_movieId')
user_emb_layer = model.get_layer("embed_new_userId")
(w_photo,) = photo_emb_layer.get_weights()
(w_user,) = user_emb_layer.get_weights()
normalize_L2(w_photo)
normalize_L2(w_user)
print(w_photo.shape, w_user.shape)

photo_ids = vle.classes_
user_ids = ule.classes_

seach_vec = np.array(w_photo, dtype=np.float32)
index_vec = np.array(w_user, dtype=np.float32)

index = faiss.IndexFlatIP(8)
index2 = faiss.IndexIDMap(index)
index2.add_with_ids(seach_vec, photo_ids)
D,I = index2.search(index_vec,10)
#print(I)
#print(D)
user_ids = user_ids.reshape(-1,1)
#print(user_ids.shape)
#print(I.shape)
final = np.concatenate((user_ids,I),axis=1) #np.concatenate((user_ids,I))
final = pd.DataFrame(final)
final.to_csv('u2i.csv',index=False)

#######################
#start FM i2i
#model.summary()

import faiss
from faiss import normalize_L2
photo_emb_layer = model.get_layer('embed_new_movieId')
user_emb_layer = model.get_layer("embed_new_userId")
(w_photo,) = photo_emb_layer.get_weights()
(w_user,) = user_emb_layer.get_weights()
normalize_L2(w_photo)
normalize_L2(w_user)
print(w_photo.shape, w_user.shape)

photo_ids = vle.classes_
user_ids = ule.classes_

seach_vec = np.array(w_photo, dtype=np.float32)
index_vec = np.array(w_photo, dtype=np.float32)

index = faiss.IndexFlatIP(8)
index2 = faiss.IndexIDMap(index)
index2.add_with_ids(seach_vec, photo_ids)
D,I = index2.search(index_vec,10)
#print(I)
#print(D)
photo_ids = photo_ids.reshape(-1,1)
#print(photo_ids.shape)
#print(I.shape)
final = np.concatenate((photo_ids,I),axis=1) #np.concatenate((user_ids,I))
final = pd.DataFrame(final)
final.to_csv('i2i.csv',index=False)
################
#let's check some example
print("reading i2i")
import pandas as pd
final = pd.read_csv('i2i.csv',header=0)
#now let check i2i results
movies = pd.read_csv("movie.csv",header=0,sep=",",index_col=None)
#movies.columns = ['id','title','Genre']
#print(movies)
#example = final.iloc[227]
example = final.iloc[2000]
print("let's check i2i,first item is")
first_item = example[0]
print(movies.loc[movies['movieId']==first_item]['title'])
print("let's check i2i,similar item are:")
for item in example[1:]:
	print(movies.loc[movies['movieId']==item]['title']) 
