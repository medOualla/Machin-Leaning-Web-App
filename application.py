import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import KFold, cross_val_score, train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif

#Pystacknet
from pystacknet.pystacknet import StackNetClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, GradientBoostingRegressor, RandomForestRegressor, ExtraTreesClassifier

st.title("Application du Machine Learing & Covid 19")


menus = ['Covid-19', 'About']
menu = st.sidebar.selectbox("Selectionner le menu", menus)


####COVID-19####
def encodage(df):
    code = {'negative': 0,
            'positive': 1,
            'not_detected': 0,
            'detected': 1
            }

    for col in df.select_dtypes('object').columns:
        df.loc[:, col] = df[col].map(code)

    return df

def imputation(df):
    df = df.dropna(axis=0)
    return df

def feature_engineering(df, viral_columns):
    df['est malade'] = df[viral_columns].sum(axis=1) >= 1
    df = df.drop(viral_columns, axis=1)
    return df

def preprocessing(df):
    X = df.drop('SARS-Cov-2 exam result', axis=1)
    y = df['SARS-Cov-2 exam result']
    return X, y

def get_Covid_19():
    pd.set_option('display.max_row', 111)
    pd.set_option('display.max_column', 111)

    data = pd.read_excel('dataset.xlsx')
    df = data.copy()

    missing_rate = df.isna().sum() / df.shape[0]
    blood_columns = list(df.columns[(missing_rate < 0.9) & (missing_rate > 0.88)])
    viral_columns = list(df.columns[(missing_rate < 0.80) & (missing_rate > 0.75)])

    important_columns = ['Patient age quantile', 'SARS-Cov-2 exam result']

    df = df[important_columns + blood_columns + viral_columns]
    df = df.reset_index()
    df= df.rename(columns={"index": "Personne Id"})
    df['index'] = df["Personne Id"].astype('int')

    trainset, testset = train_test_split(df, test_size=0.2, random_state=0)

    trainset = encodage(trainset)
    testset = encodage(testset)

    trainset = feature_engineering(trainset, viral_columns)
    testset = feature_engineering(testset, viral_columns)

    trainset = imputation(trainset)
    testset = imputation(testset)

    X_train, y_train = preprocessing(trainset)
    X_test, y_test = preprocessing(testset)
    st.dataframe(trainset.head())

    return X_train, X_test, y_train, y_test

def evaluation(model,X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    N, train_score, val_score = learning_curve(model, X_train, y_train,
                                               cv=4, scoring='f1',
                                               train_sizes=np.linspace(0.1, 1, 10))

    plt.figure(figsize=(12, 8))
    st.pyplot(N, train_score.mean(axis=1), label="Train score")
    st.pyplot(N, val_score.mean(axis=1), label="Validation score")
#####~Covid 19~#####

####Importer un fichier####
def is_binary_file(file_obj):
        return not hasattr(file_obj, 'encoding')

def get_data():
    uploaded_file_type = ['xlsx', 'csv', 'txt', 'xls']
    fichier = st.file_uploader("Telecharger un fichier", uploaded_file_type)
    if fichier is not None:
        if is_binary_file(fichier):
            data_set = pd.read_excel(fichier)
        else:
            data_set = pd.read_csv(fichier)
        return data_set
    else:
        return None
####~Importer un fichier~####

####Stacking####
def get_stacking_1(models_selection, meta_model):
    niveau_0 = list()
    for i in range(len(models_selection)):
        niveau_0.append((models_selection[i], models.get(models_selection[i])))

    niveau_1 = meta_models.get(meta_model)

    model = StackingClassifier(estimators=niveau_0, final_estimator=niveau_1, cv=5)
    return model
####~Stacking~####

if menu == 'Covid-19':
    st.sidebar.info("STACKNET MODEL")
    models_1 = dict()
    models_1["RandomFores"] = RandomForestClassifier(n_estimators=100, criterion="entropy", max_depth=5,max_features=0.5, random_state=1)
    models_1["ExtraTree"] = ExtraTreesClassifier(n_estimators=100, criterion="entropy", max_depth=5,max_features=0.5, random_state=1)
    models_1["GradiantBoosting"] = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,max_features=0.5, random_state=1)
    models_1["Logistic Regression"] = LogisticRegression(random_state=1)
    models_1["LDA"] = LinearDiscriminantAnalysis()
    models_1["Cart"] = DecisionTreeClassifier()
    models_1["Nb"] = GaussianNB()
    models_1["KNN"] = KNeighborsClassifier()
    models_1["Boosting"] = AdaBoostClassifier()
    n_niveaux = st.sidebar.number_input("Entrer la taille (Max = 5)", min_value=2, max_value=5)
    if n_niveaux:
        selections = dict()
        for i in range(0, n_niveaux):
            selections[f'Niveau {i}'] = st.sidebar.multiselect(f'Niveau {i}', list(models_1.keys()))

        st.sidebar.info("Configuration des paramaitre du stacknet")
        metric_list = ['auc', 'logloss', 'accuracy', 'f1', 'matthews']

        metrics = st.sidebar.selectbox("Metrique", metric_list)
        folds = st.sidebar.slider("Folds", 4, 12)
        restacking = st.sidebar.selectbox("Restacking", [False, True])
        use_proba = st.sidebar.selectbox("Use Proba", [True, False])
        use_retraining = st.sidebar.selectbox("Use Retraining", [True, False])
        n_jobs = st.sidebar.slider("N Jobs", 1, 10)

        param_stacknet = dict()
        param_stacknet['folds'] = folds
        param_stacknet['metric'] = metrics
        param_stacknet['n_jobs'] = n_jobs
        param_stacknet['random_state'] = 0
        param_stacknet['restacking'] = restacking
        param_stacknet['use_proba'] = use_proba
        param_stacknet['use_retraining'] = use_retraining
        param_stacknet['verbose'] = 1

        terminer = st.sidebar.checkbox("terminer")

        if (len(selections) >= 2) & terminer:
            if st.checkbox("Les modéles sélectionnées") & terminer:
                for names, models in selections.items():
                    st.text(f'{names}, {models}')

                if st.checkbox("Générer le model stacking"):
                    niveaux = dict()
                    for name, models in selections.items():
                        les_models = list()
                        for i in range(len(models)):
                            les_models.append(models_1.get(models[i]))
                        niveaux[name] = les_models

                    pystacknet_model = list()
                    for models in niveaux.values():
                        pystacknet_model.append(models)

                    model = StackNetClassifier(pystacknet_model,
                                                metric=param_stacknet["metric"],
                                                folds=param_stacknet['folds'],
                                                restacking=param_stacknet['restacking'],
                                                use_retraining=param_stacknet['use_retraining'],
                                                use_proba=param_stacknet['use_proba'],
                                                random_state=param_stacknet['random_state'],
                                                n_jobs=param_stacknet['n_jobs'],
                                                verbose=param_stacknet['verbose'])
                    if model:
                        st.info("Génération du model StackNet est terminé")

    choix = st.checkbox("Afficher le datasetCovid")
    if choix:
        X_train, X_test, y_train, y_test = get_Covid_19()
        if st.checkbox("Affichez les shape"):
            st.text(X_train.shape)

        if st.checkbox("Evaluer") & choix:
            model.fit(X_train, y_train)
            output = model.predict_proba(X_test)

            output_copy = output
            output_copy = pd.DataFrame(output_copy)
            output_copy = output_copy.reset_index()

            output_copy = output_copy.rename(index=str, columns={'index': 'Personne Id', 0: 'Negative Proba',
                                                                     1: 'Positive Proba'})

            output_copy['Personne Id'] = output_copy.replace(range(0, 111), X_test['Personne Id'])

            output_copy["Personne Id"] = output_copy["Personne Id"].astype("int")

            output_copy["Covid test"] = output_copy['Negative Proba'] < output_copy['Positive Proba']

            output_copy['Covid test'] = output_copy['Covid test'].replace([True, False], ['Positive', 'Negative'])

            output_copy = output_copy[['Personne Id', 'Negative Proba', 'Positive Proba', 'Covid test']]
            liste_des_personnes = output_copy['Personne Id'].to_list()
            tableau = output[:, 0] < output[:, 1]
            if output_copy is not None:
                st.info("Evaluation et prédiction terminé avec succés, vous pouvez voire le fichier ou vous pouvez chercher une personne.")
                output_copy.to_csv("covid19resu.csv", index=False, header=True)
                personne = st.multiselect("Chercher l'id", liste_des_personnes)
                if personne:
                    st.success("Personne trouvé, voila les resultats")
                    recherche_resultat = output_copy[output_copy['Personne Id'] == personne[0]]
                    st.dataframe(recherche_resultat)
                if st.checkbox("Afficher la table de confusion..."):
                    st.text(confusion_matrix(y_test, tableau))
            else:
                st.error("Erreur Essayer plus tard...")
