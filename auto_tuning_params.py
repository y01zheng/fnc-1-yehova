import sys
import numpy as np
import csv
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features, sentiment_features
from tf_idf_feature import *
from utils.dataset import DataSet
from utils.generate_test_splits import generate_hold_out_split,read_ids
from utils.score import report_score, LABELS, score_submission
from utils.system import parse_params, check_version

lim_unigram = 5000

def training_holdout_split(dataset, training = 0.8, base_dir="splits"):
    if not (os.path.exists(base_dir+ "/"+ "training_ids.txt")
            and os.path.exists(base_dir+ "/"+ "hold_out_ids.txt")):
        generate_hold_out_split(dataset,training,base_dir)

    training_ids = read_ids("training_ids.txt", base_dir)
    hold_out_ids = read_ids("hold_out_ids.txt", base_dir)

    return training_ids, hold_out_ids


def get_stances_for_train_and_holdout(dataset,train,hold_out):
    stances_train = []
    stances_hold_out = []
    for stance in dataset.stances:
        if stance['Body ID'] in hold_out:
            stances_hold_out.append(stance)
        else:
            stances_train.append(stance)
    return stances_train,stances_hold_out

def generate_features(stances,dataset,name,bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])


    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    
    # X_refuting_body: we add a new refuting feature about the existence of refuting words in the body
    X_refuting_head, X_refuting_body = refuting_features(h, b)
    
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")
    
    # X_senti_head: The sentiment vector of the headline;
    # X_senti_body:  The sentiment vector of the body;
    # X_senti_cos : The cosine similarity between the sentiment vectors of the headline and body.

    X_senti_head,X_senti_body,X_senti_cos = sentiment_features(h, b)

    # X_tf_cos : The cosine similarity between the TF vectors of the headline and body
    # X_tf_idf_cos : The cosine similarity between the TF-IDF vectors of the headline and body.
    X_tf_cos, X_tf_idf_cos = gen_tf_idf_feats(stances,dataset.articles,bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

    X = np.c_[X_hand, X_polarity, X_refuting_head,X_refuting_body, X_overlap, X_senti_head,X_senti_body,X_senti_cos,X_tf_cos,X_tf_idf_cos]
    return X,y

if __name__ == "__main__":
    check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet()
    training,hold_out = training_holdout_split(d)
    train_stances, hold_out_stances = get_stances_for_train_and_holdout(d,training,hold_out)

    # bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = tf_idf_preprocess(d, competition_dataset, lim_unigram=lim_unigram)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = tf_idf_preprocess(d, competition_dataset, lim_unigram=lim_unigram)

    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition",bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

    X_training,y_training = generate_features(train_stances,d,"train",bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout",bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
           
    params = {'n_estimators':[10,40,90,120,150,170,200],'learning_rate':[0.01, 0.02, 0.05, 0.1],'max_depth':[2,3,4,5]}
    gbc = GradientBoostingClassifier(random_state=14128, verbose=True)
    clf = GridSearchCV(gbc, params, cv=10)
    clf.fit(X_training, y_training)
    
    print(clf.best_score_)
    print(clf.best_params_)
        
