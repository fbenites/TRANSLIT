
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC

import numpy as np


#prepare data

def transform_features(data,  cv=None):

    import jellyfish
    import zlib
    #import nltk
    #arpabet = nltk.corpus.cmudict.dict()
    lefts = []
    rights = []
    labels = []
    X_ed=[]

    
    for t1 in data:
        if len(t1)==3:
            n1, n2, label = t1
        else:
            n1, n2, label, *rest =t1
        lefts.append(n1)
        rights.append(n2)
        X_ed.append([jellyfish.jaro_distance(n1,n2), jellyfish.levenshtein_distance(n1,n2)/float(max(len(n1),len(n2))),
            len(zlib.compress((n1+n2).encode("utf-8")))/float(len(zlib.compress((n1).encode("utf-8")))+
                               len(zlib.compress(n2.encode("utf-8")))),
            ])
        labels.append(label)
        

    X = cv.transform(lefts) - cv.transform(rights)
    import scipy.sparse
    X = scipy.sparse.hstack([X,X_ed]).tocsc()
    return X, np.array(labels)


# create data generator







def do_classification(X_train_raw, X_dev_raw, y_train, y_dev, gen_pars):
    scores=[]
    data_train = [gen_pars[tk] for tk in X_train_raw]

    data_dev = [gen_pars[tk] for tk in X_dev_raw]

    #tf_train = tf_vectorizer.fit_transform([data[0]])
    #tf_dev = tf_vectorizer.transform(sents_dev)



    cv = CountVectorizer(
                analyzer="char_wb",
                preprocessor=lambda x : x)
    data_train_cv = [tj for tk in data_train for tj in tk[:2]]
    cv.fit(data_train_cv)
    X_train, y_train_l1 = transform_features(data_train, cv=cv)
    X_test, y_test = transform_features(data_dev, cv=cv)

    rf = RandomForestClassifier(
                n_estimators=100,random_state=42,
    bootstrap=False, n_jobs=16)

    print("X_train",X_train.shape, X_test.shape)
        #
    rf.fit(X_train,y_train)
    print("RF whole features",rf.score(X_test,y_test))
    scores.append(rf.score(X_test,y_test))
    

    clf_svm = LinearSVC(random_state=0,C=1)

    clf_svm.fit(X_train,y_train)
    print("SVM whole features",clf_svm.score(X_test,y_test))
    scores.append(clf_svm.score(X_test,y_test))



    data_train_s = [ tk[0].split("_")[1]+" "+tk[1].split("_")[1] for tk in data_train]
    data_dev_s = [ tk[0].split("_")[1]+" "+tk[1].split("_")[1] for tk in data_dev]

    from sklearn.feature_extraction.text import TfidfVectorizer

    
    tfidf_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1,3))
    x_train_tf = tfidf_vec.fit_transform(data_train_s)
    x_dev_tf = tfidf_vec.transform(data_dev_s)

    clf_svm = LinearSVC(random_state=0,C=1)

    clf_svm.fit(x_train_tf,y_train)
    print("SVM tf-idf char",clf_svm.score(x_dev_tf,y_test))
    scores.append(clf_svm.score(x_dev_tf,y_test))
    return scores




if __name__ == "__main__":
    #load pairs
    if "gen_pars" not in globals():
        import pickle
        gen_pars = pickle.load(open("../artefacts/gen_pairs.pkl","rb"))


    #Tab 4
    idxs = range(len(gen_pars))
    X_train_raw_t, X_dev_raw_t, y_train_t, y_dev_t = train_test_split(idxs,[tk[2] for tk in gen_pars],test_size=0.2,random_state=42)
    do_classification(X_train_raw_t, X_dev_raw_t, y_train_t, y_dev_t, gen_pars)

    print("CV")
    results_cv_tab4 = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(idxs):
        X_train_raw_t = train_index
        X_dev_raw_t = test_index
        y_train_t = [gen_pars[tk][2] for tk in train_index]
        y_dev_t = [gen_pars[tk][2] for tk in test_index]
        results_cv_tab4.append(do_classification(X_train_raw_t, X_dev_raw_t, y_train_t, y_dev_t, gen_pars))

    ## Experiment simple, first 100'000 positive, then 100'000 negative, then difficult neg

    idxs = range(200000)


    print("SIMPLE")
    X_train_raw_SIMPLE, X_dev_raw_SIMPLE, y_train_SIMPLE, y_dev_SIMPLE = train_test_split(idxs,[tk[2] for tk in gen_pars[:200000]],test_size=0.2,random_state=42)
    do_classification(X_train_raw_SIMPLE, X_dev_raw_SIMPLE, y_train_SIMPLE, y_dev_SIMPLE, gen_pars)


    print("CV")
    results_cv_tab3 = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(idxs):
        X_train_raw_t = train_index
        X_dev_raw_t = test_index
        y_train_t = [gen_pars[tk][2] for tk in train_index]
        y_dev_t = [gen_pars[tk][2] for tk in test_index]
        results_cv_tab3.append(do_classification(X_train_raw_t, X_dev_raw_t, y_train_t, y_dev_t, gen_pars))


    # use simaese with angular distance
    # https://github.com/Samurais/deep-siamese-text-similarity

    #use cnn siamese
    #https://github.com/PhyloStar/SiameseConvNet

    X_train_raw = X_train_raw_t
    X_dev_raw = X_dev_raw_t

    X_train_N = [ "##".join([tj.split("_")[1] for tj in gen_pars[tk][:2]]) for tk in X_train_raw]
    X_test_N = [ "##".join([tj.split("_")[1] for tj in gen_pars[tk][:2]]) for tk in X_dev_raw]
    # use var-autoencoder

    # use bert...



    #import spacy
    #import torch
    #import numpy
    from numpy.testing import assert_almost_equal

    #is_using_gpu = spacy.prefer_gpu()
    #if is_using_gpu:
    #        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    #nlp = spacy.load("en_trf_bertbaseuncased_lg")

    #x_train_b = [nlp(gen_pars[tk][0].split("_")[1]).similarity(npl(gen_pars[tk][1].split("_")[1])) for tk in X_train_raw_t]
    #x_dev_b = [nlp(gen_pars[tk][0].split("_")[1]).similarity(nlp(gen_pars[tk][1].split("_")[1])) for tk in X_dev_raw_t]
