import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features,sentiment_features
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
    
    X_refuting_head, X_refuting_body = refuting_features(h, b)
    
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X_senti_head,X_senti_body,X_senti_cos = sentiment_features(h, b)

    X_tf_cos, X_tf_idf_cos = gen_tf_idf_feats(stances,dataset.articles,bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
   
    X = np.c_[X_hand, X_polarity, X_refuting_head,X_refuting_body, X_overlap,X_tf_cos,X_tf_idf_cos]
    return X,y

if __name__ == "__main__":
    check_version()
  
    n_iter = 200
    learning_rate = 0.01
    max_depth = 3

    if len(sys.argv)==4:
        n_iter = int(sys.argv[1])
        learning_rate = float(sys.argv[2])
        max_depth = int(sys.argv[3])

    #Load the training dataset and generate folds
    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    # bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = tf_idf_preprocess(d, competition_dataset, lim_unigram=lim_unigram)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = tf_idf_preprocess(d, competition_dataset, lim_unigram=lim_unigram)

    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition",bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

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

        clf = GradientBoostingClassifier(n_estimators=n_iter,learning_rate=learning_rate,max_depth=max_depth, random_state=14128, verbose=True)
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
    dev_accu = report_score(actual,predicted)
    print("")
    print("")
 
    with open("results_dev" + str(max_depth) +".txt", 'a+') as f:
        f.write("n_itr," + str(n_iter) + ",learning_rate,"+str(learning_rate)+",score,"+str(dev_accu)+"\n")

    #Run on competition dataset
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    accuracy = report_score(actual,predicted)
    with open("results"+ str(max_depth) +".txt", 'a+') as f:
        f.write("n_itr," + str(n_iter) + ",learning_rate,"+str(learning_rate)+",score,"+str(accuracy)+"\n")
