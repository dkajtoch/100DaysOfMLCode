from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import sys

# ---------------------------
# read already prepared data
# ---------------------------
data_train = pd.read_csv('./data/data_train.csv')
y = data_train['renewal']
X = data_train.drop('renewal', axis=1)

# ----------------------------
# Define model
# ----------------------------
input_dim = len(X.columns)

model = Sequential()
model.add(Dense(32, input_dim=input_dim, kernel_initializer='random_uniform', activation='relu'))
model.add(Dense(16, kernel_initializer='random_uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='random_uniform', activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# combine scaler with neural network
clf = Pipeline([('feature', StandardScaler()), ('model', model)])

if str(sys.argv[1]) == 'tuning':
    print('Preparing for hyperparameter tuning')
    print('No tuning available')

elif str(sys.argv[1]) == 'testing':
    print('Preparing for testing')

    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, log_loss
    import random
    from tqdm import tqdm

    # -----------------------------------------------
    # evaluate model performance for different sets
    # -----------------------------------------------
    iterations = 1
    n_splits = 2

    # progress bar
    ProgressBar = tqdm(range(iterations*n_splits))

    loss_score = []
    auc_score = []
    for i in range(0,iterations):
        # Split the data
        r_state = random.randint(1,1000000)
        skf = StratifiedKFold(n_splits=n_splits, random_state=r_state, shuffle=True)

        for train_index, test_index in skf.split(X,y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            clf.fit(X_train, y_train,
                    model__epochs=50,
                    model__batch_size=10,
                    model__verbose=True
                   )

            ProgressBar.update()

            prob = np.float64(clf.predict_proba(X_test))

            loss_score.append(log_loss(y_test, prob))
            auc_score.append(roc_auc_score(y_test, prob))

    loss_score = np.array(loss_score)
    auc_score  = np.array(auc_score)
    # -------------------------
    # Print results
    # -------------------------
    print( 'AUC: %.5f +/- %.5f' % (np.mean(auc_score), np.std(auc_score)) )
    print( 'Loss: %.5f +/- %.5f' % (np.mean(loss_score), np.std(loss_score)) )

else str(sys.argv[1] == 'predict'):
    print('Preparing for prediction')

    clf.fit(X, y,
            epochs=50,
            batch_size=10,
            verbose=True
           )

    # read test data
    X_test = pd.read_csv('./data/data_test.csv')
    proba = np.float64(clf.predict_proba(X_test))

    # export to a file
    export_data = pd.read_csv('./data/test.csv', names=['id'])
    export_data['renewal'] = proba
    export_data['premium'] = X_test['premium']

    export_data.to_csv('./data/test_proba.csv', index=False)
