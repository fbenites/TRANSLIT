import keras

from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Embedding, Lambda
from keras import backend as K
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical

from keras.preprocessing import text

from keras.preprocessing import sequence

from sklearn.model_selection import KFold
import numpy as np


MAX_SEQ_LENGTH=200
MAX_VOCAB_SIZE=100

def sequentialize_data(train_contents, val_contents=None):
    """Vectorize data into ngram vectors.
    Args:
        train_contents: training instances
        val_contents: validation instances
        y_train: labels of train data.
    Returns:
        sparse ngram vectors of train, valid text inputs.
    """
    tokenizer = text.Tokenizer(num_words = MAX_VOCAB_SIZE, char_level=True)
    tokenizer.fit_on_texts(train_contents)
    x_train = tokenizer.texts_to_sequences(train_contents)

    if val_contents:
        x_val = tokenizer.texts_to_sequences(val_contents)

    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQ_LENGTH:
        max_length = MAX_SEQ_LENGTH

    x_train = sequence.pad_sequences(x_train,padding='post', maxlen=max_length)
    if val_contents:
        x_val = sequence.pad_sequences(x_val, padding='post',maxlen=max_length)

    word_index = tokenizer.word_index
    num_features = min(len(word_index) + 1, MAX_VOCAB_SIZE)
    if val_contents:
        return x_train, x_val, word_index, num_features, tokenizer, max_length
    else:
        return x_train, word_index, num_features, tokenizer, max_length


def get_siamese_model(input_shape, input_length, num_features):
    """
        Model architecture
    """

    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network
    model = Sequential()
    filters=32
    kernel_size=4
    dropout_rate=0.15
    model.add(Embedding(input_dim=num_features, output_dim=10, input_length=input_length, input_shape=input_shape))
    model.add(Conv1D(filters=2*filters, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
#    model.add(Dense(250, activation='relu'))
#    model.add(Dropout(rate=dropout_rate))
#    model.add(Dense(100, activation='relu'))
#    model.add(Dropout(rate=dropout_rate))
#    model.add(Dense(op_units, activation='sigmoid'))
    # model.add(Embedding(input_dim=num_features, output_dim=10, input_length=input_shape))

    # model.add(Conv1D(64, (10,10), activation='relu', input_shape=input_shape,
    #                  kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    # model.add(MaxPooling2D())
    # model.add(Conv2D(128, (7,7), activation='relu',
    #                  kernel_initializer=initialize_weights,
    #                  bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    # model.add(MaxPooling2D())
    # model.add(Conv2D(128, (4,4), activation='relu', kernel_initializer=initialize_weights,
    #                  bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    # model.add(MaxPooling2D())
    # model.add(Conv2D(256, (4,4), activation='relu', kernel_initializer=initialize_weights,
    #                  bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    #model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                    #kernel_regularizer=l2(1e-3),
                    #kernel_initializer=initialize_weights,bias_initializer=initialize_bias
    ))
    # Generate the encodings (feature vectors) for the two images
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid'#,bias_initializer=initialize_bias
    )(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    siamese_net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    siamese_net.summary()
    # return the model
    return siamese_net


def lstm_mode(input_shape,num_features):
    model = Sequential()
    model.add(Embedding(input_dim=num_features, output_dim=10, input_length=input_shape, input_shape=input_shape))
    
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
        
    
              

def cnn_model(input_shape,
              num_features,
              blocks=3,
              filters=64,
              kernel_size=4,
              op_units=2,
              dropout_rate=0.25):
    #op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = Sequential()
    model.add(Embedding(input_dim=num_features, output_dim=10, input_length=input_shape, input_shape=input_shape))
    model.add(Conv1D(filters=2*filters, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


if __name__=="__main__":
    #load pairs
    
    if "gen_pars" not in globals():
        import pickle
        gen_pars = pickle.load(open("../artefacts/gen_pairs.pkl","rb"))


    #Tab 4
    idxs = range(len(gen_pars))
    X_train_raw, X_dev_raw, y_train, y_dev = train_test_split(idxs,[tk[2] for tk in gen_pars],test_size=0.2,random_state=42)
    #do_classification(X_train_raw_t, X_dev_raw_t, y_train_t, y_dev_t, gen_pars)

    if "do_train" not in globals():
        do_train=0
    if do_train<1 :
        X_train_N1, X_train_N2 = [gen_pars[tk][0].split("_")[1] for tk in X_train_raw],\
                                 [gen_pars[tk][1].split("_")[1] for tk in X_train_raw]
        X_dev_N1, X_dev_N2 = [gen_pars[tk][0].split("_")[1] for tk in X_dev_raw],\
                             [gen_pars[tk][1].split("_")[1] for tk in X_dev_raw]
        x_train_s, x_val_s, word_index, num_features, tokenizer, max_length = sequentialize_data(X_train_N1, val_contents=X_dev_N1)

        x_train_s1, x_val_s1, word_index, num_features, tokenizer, max_length = sequentialize_data(X_train_N2, val_contents=X_dev_N2)

        x_train_s2, x_val_s2, word_index, num_features, tokenizer, max_length = sequentialize_data(X_train_N2, val_contents=X_dev_N2)
        if x_train_s.shape[1]>x_train_s2.shape[1]:
            x_train_s_p = x_train_s
            x_train_s2_p = sequence.pad_sequences(x_train_s2,padding='post', maxlen=x_train_s.shape[1])

        else:
            x_train_s_p = sequence.pad_sequences(x_train_s,padding='post', maxlen=x_train_s2.shape[1])
            x_train_s2_p = x_train_s2

        x_val_s_p = sequence.pad_sequences(x_val_s,padding='post', maxlen=x_train_s.shape[1])
        x_val_s2_p = sequence.pad_sequences(x_val_s2,padding='post', maxlen=x_train_s.shape[1])

        model=get_siamese_model((x_train_s_p.shape[1],),x_train_s_p.shape[1], 100)
        model.fit([x_train_s_p,x_train_s2_p],y_train, epochs=15, validation_data=([x_val_s_p, x_val_s2_p],y_dev))
        te_score = model.predict([x_val_s_p, x_val_s2_p], verbose=1)
        print("accuracy",np.where((te_score>0.5).squeeze()==np.array(y_dev))[0].shape[0]/len(y_dev))

    if do_train<2 :
        X_train_N12 = [gen_pars[tk][0].split("_")[1]+"####"+gen_pars[tk][1].split("_")[1] for tk in X_train_raw]
        X_dev_N12 = [gen_pars[tk][0].split("_")[1]+"####"+gen_pars[tk][1].split("_")[1] for tk in X_dev_raw]
        x_train12_s, x_val12_s, word_index, num_features, tokenizer, max_length = sequentialize_data(X_train_N12, val_contents=X_dev_N12)
        max_length=200
        max_feature=x_train12_s.shape[1]
        #model=get_siamese_model((max_feature,),max_feature, 200)

        #model.fit(x_train12_s,y_train)
        #te_score = model.predict(x_val12_s_p, verbose=1)
        #print("accuracy together",np.where((te_score>0.5).squeeze()==y_dev)[0].shape/len(y_dev))


        model=cnn_model((max_feature,),max_feature, 200)


        model.fit(x_train12_s,y_train, epochs=15, validation_data=(x_val12_s,y_dev))
        te_score = model.predict(x_val12_s, verbose=1)

        print("accuracy cnn",np.where((te_score>0.5).squeeze()==np.array(y_dev))[0].shape[0]/len(y_dev))
        #print("accuracy cnn",np.where((te_score>0.5).squeeze()==y_dev)[0].shape/len(y_dev))


    #cv
    idxs = range(len(gen_pars))
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores_siam_1=[]
    scores_siam_2=[]
    scores_cnn_1=[]
    for X_train_raw, X_dev_raw in kf.split(idxs):


        #X_train_raw, X_dev_raw, y_train, y_dev = train_test_split(idxs,[tk[2] for tk in gen_pars],test_size=0.2,random_state=42)
        y_train = [gen_pars[tk][2] for tk in X_train_raw]
        y_dev = [gen_pars[tk][2] for tk in X_dev_raw]
        #do_classification(X_train_raw_t, X_dev_raw_t, y_train_t, y_dev_t, gen_pars)


        X_train_N1, X_train_N2 = [gen_pars[tk][0].split("_")[1] for tk in X_train_raw],\
                                 [gen_pars[tk][1].split("_")[1] for tk in X_train_raw]
        X_dev_N1, X_dev_N2 = [gen_pars[tk][0].split("_")[1] for tk in X_dev_raw],\
                             [gen_pars[tk][1].split("_")[1] for tk in X_dev_raw]
        x_train_s, x_val_s, word_index, num_features, tokenizer, max_length = sequentialize_data(X_train_N1, val_contents=X_dev_N1)

        x_train_s1, x_val_s1, word_index, num_features, tokenizer, max_length = sequentialize_data(X_train_N2, val_contents=X_dev_N2)

        x_train_s2, x_val_s2, word_index, num_features, tokenizer, max_length = sequentialize_data(X_train_N2, val_contents=X_dev_N2)
        if x_train_s.shape[1]>x_train_s2.shape[1]:
            x_train_s_p = x_train_s
            x_train_s2_p = sequence.pad_sequences(x_train_s2,padding='post', maxlen=x_train_s.shape[1])

        else:
            x_train_s_p = sequence.pad_sequences(x_train_s,padding='post', maxlen=x_train_s2.shape[1])
            x_train_s2_p = x_train_s2

        x_val_s_p = sequence.pad_sequences(x_val_s,padding='post', maxlen=x_train_s_p.shape[1])
        x_val_s2_p = sequence.pad_sequences(x_val_s2,padding='post', maxlen=x_train_s_p.shape[1])

        model=get_siamese_model((x_train_s_p.shape[1],),x_train_s_p.shape[1], 100)
        model.fit([x_train_s_p,x_train_s2_p],y_train, epochs=15, validation_data=([x_val_s_p, x_val_s2_p],y_dev))
        te_score = model.predict([x_val_s_p, x_val_s2_p], verbose=1)
        
        print("accuracy",np.where((te_score>0.5).squeeze()==np.array(y_dev))[0].shape[0]/len(y_dev))
        scores_siam_1.append(np.where((te_score>0.5).squeeze()==np.array(y_dev))[0].shape[0]/len(y_dev))

        X_train_N12 = [gen_pars[tk][0].split("_")[1]+"####"+gen_pars[tk][1].split("_")[1] for tk in X_train_raw]
        X_dev_N12 = [gen_pars[tk][0].split("_")[1]+"####"+gen_pars[tk][1].split("_")[1] for tk in X_dev_raw]
        x_train12_s, x_val12_s, word_index, num_features, tokenizer, max_length = sequentialize_data(X_train_N12, val_contents=X_dev_N12)
        max_length=200
        max_feature=x_train12_s.shape[1]
        #model=get_siamese_model((max_feature,),max_feature, 200)

        #model.fit(x_train12_s,y_train)
        #te_score = model.predict(x_val12_s_p, verbose=1)
        #print("accuracy together",np.where((te_score>0.5).squeeze()==y_dev)[0].shape/len(y_dev))


        model=cnn_model((max_feature,),max_feature, 200)


        model.fit(x_train12_s,y_train, epochs=15, validation_data=(x_val12_s,y_dev))
        te_score = model.predict(x_val12_s, verbose=1)
        print("accuracy cnn",np.where((te_score>0.5).squeeze()==np.array(y_dev))[0].shape[0]/len(y_dev))
        scores_cnn_1.append(np.where((te_score>0.5).squeeze()==np.array(y_dev))[0].shape[0]/len(y_dev))

