import sys
import numpy as np
import csv
from sklearn.ensemble import GradientBoostingClassifier
sys.path.append("..")
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features, sentiment_features
from tf_idf_feature import *
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from utils.system import parse_params, check_version

lim_unigram = 5000

def generate_features(stances,dataset,name,bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])


    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    
    # X_refuting_body: I add a new refuting feature about the existence of refuting words in the body
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

    X = np.c_[X_hand, X_polarity, X_refuting_head, X_overlap, X_tf_idf_cos]
    return X,y

if __name__ == "__main__":
    check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet(name="train", path="./../fnc-1")
    folds,hold_out = kfold_split(d,n_folds=10,base_dir="./../splits")
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    # bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = tf_idf_preprocess(d, competition_dataset, lim_unigram=lim_unigram)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test",path="./../fnc-1")
    bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = tf_idf_preprocess(d, competition_dataset, lim_unigram=lim_unigram)

    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition",bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

    headline,bodyId,stance = [],[],[]
    for s in competition_dataset.stances:
        headline.append(s['Headline'])
        bodyId.append(s["Body ID"])

    Xs = dict()
    ys = dict()


    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout",bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold),bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)


    best_score = 0
    best_fold = None


    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]  #delete ids related to current fold 

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        clf = GradientBoostingClassifier(n_estimators=200,learning_rate=0.05, random_state=14128, verbose=True)
        clf.fit(X_train, y_train)

        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf



    #Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(actual,predicted)
    print("")
    print("")

    #Run on competition dataset
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    actual = [LABELS[int(a)] for a in y_competition]
            
    print("Scores on the test set")
    report_score(actual,predicted)
