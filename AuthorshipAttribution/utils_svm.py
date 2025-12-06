# Funzioni utili per l'implementazione delle task che utilizzano il classificatore SVM. Le funzioni presenti nei notebook sono specifiche
# alla task espletata nel notebook, le funzioni in questo file sono usate in più notebooks.
# Eccezion fatta per alcune funzioni specifiche alla task numero 2 che sono presenti in questo script per non affollare troppo il notebook
# queste sono get_num_features, filter_features, training_val

# Import dei moduli
import numpy as np
import json

from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction import DictVectorizer

# Funzione per il loading del dizionario
def load_data(path):

    """
    Funzione atta a caricare un dizionario. 
    Prende in input il percorso del file da leggere e ritorna il dizionario.

    :args path: str, il percorso del file da caricare
    :returns data: dic instance
    """

    with open(path, 'r', encoding = 'UTF-8') as src_file:
        data = json.load(src_file)
    return data


# Funzione per lo split in training, validation e test set
def train_test_split(data, target_label):

    """
    Funzione per la divisione in train, validation e test set di ogni istanza con le rispettive features.
    
    :args data: oggetto dizionario contenente ogni record di train, validation e test
    :args: target_label: la chiave utilizzata come etichetta target
    
    :returns train_features, train_labels: liste di valori che saranno usate come set di training contenenti rispettivamente le features di Profiling UD identificate per ogni istanza e l'etichetta author associata
    :returns val_features, val_labels: liste di valori che saranno usate come set di validazione contenenti rispettivamente le features di Profiling UD identificate per ogni istanza e l'etichetta author associata
    :returns test_features, test_labels: liste di valori che saranno usate come set di test contenenti rispettivamente le features di Profiling UD identificate per ogni istanza e l'etichetta author associata
    """

    train_features, val_features, test_features = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    for doc_id in data: 
        try:
            split = data[doc_id]['split']
            features = data[doc_id]['features']
            label = data[doc_id][target_label]
        except KeyError as e:
            print(f"Errore KeyError {e} per doc_id: {doc_id}")
            continue

        if split == 'train':
            train_features.append(features)
            train_labels.append(label)

        elif split == 'val':
            val_features.append(features)
            val_labels.append(label)

        elif split == 'test':
            test_features.append(features)
            test_labels.append(label)

        else: 
            print(f"Attenzione: split non riconosciuto per {doc_id}: {split}")

    return train_features, train_labels, val_features, val_labels, test_features, test_labels


# Funczione per recuperare il numero di features
def get_num_features(features_dict):
    """
    Funzione che legge l'attributo features di un dizionario e lo inserisce all'interno di un set di cui poi ritorna la lunghezza.

    :args features_dict: dizionario con le features

    :returns len(all_features): il numero totale di features trovate
    
    """
    all_features = set()
    for document_feats in features_dict:
        all_features.update(list(document_feats.keys()))
        return len(all_features)


# Funzione usata per filtrare le features sulla base della frequenza
def filter_features(train_features, min_occurrences):

    """
    Funzione che filtra le features nei dizionari dei documenti, rimuovendo quelle che compaiono meno di una soglia minima specificata.

    :args train_features: lista di dizionari, ciascuno contenente le features di un documento
    :args min_occurrences: intero che rappresenta il numero minimo di occorrenze richiesto affinché una feature venga mantenuta

    :returns train_features: la stessa lista di dizionari, ma con le features rare rimosse
    """

    features_counter = dict()
    for document_features_dict in train_features:
        for feature in document_features_dict:
            if feature in features_counter:
                features_counter[feature] +=1
            else:
                features_counter[feature] = 1

    for document_features_dict in train_features:
        document_features = list(document_features_dict.keys())

        for feature in document_features:
            if features_counter[feature] < min_occurrences:
                document_features_dict.pop(feature)

    return train_features


# Funzione usata per selezionare i migliori parametri per il modello
def model_selection(x, y):

    """
    Funzione che prende in input un array di features ed uno di etichette, che utilizza per cercare i migliori iperparametri del
    modello usando una grid search con kfold = 5.

    :param x: array di features 
    :param y: array di etichette 

    :returns grid_results: i risultati della grid search
    """

    clf = LinearSVC(max_iter=50000)

    SVC_params = {
        'dual': [False], # Non proviamo dual = True perché ottimizzato per dataset con più campioni che features
        'C' : [0.01, 0.1, 1, 10, 100],
        'class_weight' : ['balanced', None],
        'penalty' : ['l1', 'l2'],
        'fit_intercept' : [True, False]
    }

    folds = KFold(n_splits = 5, shuffle = True, random_state= 42)

    SVC_grid = GridSearchCV(estimator= clf, 
                            param_grid= SVC_params, 
                            cv = folds, 
                            n_jobs = -1, 
                            scoring= 'accuracy', 
                            return_train_score= True, 
                            verbose= 1)
    
    grid_results = SVC_grid.fit(x, y)
    print(grid_results.best_score_)


    return grid_results


# Funziona usata per predirre da un test set
def predict (clf, x_test, y_test, split):

    """
    Funzione che prende in input un classificatore, un vettore di features ed un vettore di etichette e la split sulla quale stiamo
    effettuando la predizione, e stampa l'accuracy, il classification report e la confusio matrix della predizione.

    :arg clf: il classificatore da usare per la predizione
    :arg x_test: vettore di features
    :arg y_test: vettore di etichette
    :arg split: lo split su cui si sta effettuando la predizione

    :returns y_pred: le predizioni effettuate dal classificatore
    """

    y_pred = clf.predict(x_test)

    print(f"Accuracy sul {split}: {accuracy_score(y_test, y_pred)}")

    print(f"Classification report sul {split}: {classification_report(y_test, y_pred, zero_division= 0)}")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation='vertical')

    return y_pred


# Funzione usata per automatizzare il training e la validation per tutte le features estratte nel task 2
def training_val(all_docs, min_occurrences):

    """
    Funzione che esegue l'intero processo di training: pulizia delle features, vettorizzazione, scalatura, 
    selezione del modello e valutazione su validation set.

    :args all_docs: lista di documenti, ciascuno rappresentato come una tupla (features_dict, label)
    :args min_occurrences: intero che indica la soglia minima di occorrenze richieste per mantenere una feature

    :returns x_test: features del test set trasformate e scalate
    :returns y_test: etichette del test set
    :returns model: modello LinearSVC addestrato con i migliori iperparametri trovati
    """

    # Divisione dei documenti in train, validation and test set
    train_features, train_labels, val_features, val_labels, test_features, test_labels = train_test_split(all_docs)

    # Pulizia delle features mantenendo solo quelle che hanno un'occorrenza minima prestabilita
    print('Features pre cleaning: ', len(train_features))
    train_features = filter_features(train_features, min_occurrences)
    print(f'Numero features post filtro: {get_num_features(train_features)}')

    # Trasformazione in vettore e scalatura delle features
    vectorizer = DictVectorizer()
    scaler = MaxAbsScaler()

    x_train = scaler.fit_transform(vectorizer.fit_transform(train_features))
    y_train = np.array(train_labels)

    x_val = scaler.transform(vectorizer.transform(val_features))
    y_val = np.array(val_labels)

    x_test = scaler.transform(vectorizer.transform(test_features))
    y_test = np.array(test_labels)

    # Model Selection
    grid_results = model_selection(x_train, y_train)
    model = LinearSVC(**grid_results.best_params_)
    model.fit(x_train, y_train)

    # Assessment sul set di validazione
    predict(model, x_val, y_val, split = 'validation set')

    dummy_clf = DummyClassifier(strategy="uniform")
    dummy_clf.fit(x_train, y_train)
    print(f"Accuracy del DummyClassifier: {dummy_clf.score(x_val, y_val)}")

    return x_test, y_test, model