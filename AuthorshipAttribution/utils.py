import os 
import pandas as pd
import re
import nltk
from loguru import logger
nltk.download('punkt_tab')

def abs_path(relative_path_in_data, data_root="data"):
    
    """
    Restituisce il percorso assoluto di un file a partire da un percorso relativo all'interno di una directory dati.

    La funzione costruisce il percorso assoluto combinando la directory corrente del notebook,
    la directory dei dati specificata (di default 'data') e il percorso relativo fornito.

    Args:
        relative_path_in_data (str): Percorso relativo del file all'interno della directory dei dati.
        data_root (str, optional): Nome della directory principale contenente i dati. Default: "data".

    Returns:
        str: Percorso assoluto del file.
    """

    notebook_dir = os.getcwd()  # directory corrente da cui Jupyter viene eseguito
    data_file_path = os.path.join(notebook_dir, data_root, relative_path_in_data)
    return os.path.abspath(data_file_path)


def read_file(file_path):
    
    """
    Apre il file in modalità lettura (`"r"`) con codifica UTF-8 e ne restituisce l'intero contenuto.

    Args:
        file_path (str): Percorso del file da leggere.

    Returns:
        str: Contenuto del file come stringa.
    """
  
    with open (file_path, "r", encoding= "utf-8") as infile:
        contents = infile.read()
    return contents 


def strip_to_text(start_match, end_match, data):

    """
    Estrae e restituisce il testo compreso tra due match (pattern trovati) all'interno di una stringa, rimuovendo eventuali spazi bianchi iniziali e finali.

    La funzione utilizza gli oggetti di tipo `re.Match` forniti per determinare l'intervallo di testo da estrarre. Se uno dei due match è assente (None),
    viene sollevata un'eccezione.

    Args:
        start_match (re.Match): Oggetto `Match` che indica l'inizio della sezione da estrarre.
        end_match (re.Match): Oggetto `Match` che indica la fine della sezione da estrarre.
        data (str): Testo sorgente da cui estrarre la sottostringa.

    Returns:
        str: Testo compreso tra i due match, con spazi bianchi rimossi dai bordi.

    Raises:
        ValueError: Se uno dei match (`start_match` o `end_match`) è assente.
    """
    
    # Controlla che la stringa iniziale e finale siano presenti
    if start_match and end_match:
        # Mantiene solo quello che si trova tra le due stringhe
        start_index = start_match.start()
        end_index = end_match.start()
        main_text = data[start_index:end_index].strip()
    
    else:
        raise ValueError("I termini non esistono nel testo")
    
    return main_text


def paragraphs_sep(text):

    """
    Suddivide un testo in paragrafi, separando le sezioni delimitate da una o più righe vuote.

    La funzione utilizza una regex per dividere il testo ogni volta che trova una o più righe vuote,
    rimuove eventuali spazi bianchi superflui da ciascun paragrafo, ed esclude paragrafi vuoti dal risultato.

    Args:
        text (str): Il testo da suddividere in paragrafi.

    Returns:
        list of str: Elenco dei paragrafi non vuoti, ciascuno come stringa pulita.
    """
    
    # Dividiamo il text in paragrafi, cioè blocchi di testo separati da una o più righe vuote.
    paragraphs = re.split(r'\n\s*\n', text)
    # Puliamo ogni paragrafo ed elimina eventuali stringhe vuote.
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return paragraphs


def cleaning(text):

    """
    La funzione esegue le seguenti operazioni su ciascun elemento della lista:
    - Rimuove spazi vuoti iniziali e finali.
    - Elimina interruzioni di riga (`\n`).
    - Rimuove eventuali intestazioni come "CHAPTER 1", "Chapter II", ecc.
    - Riduce spazi multipli a uno solo.
    - Esclude paragrafi vuoti dal risultato finale.

    :args text (list of str): Lista di stringhe/paragrafi da pulire.

    :returns list of str: Lista di paragrafi puliti.
    """
    # Inizilizziamo una lista vuota per contenere i paragrafi puliti
    paragraphs_clean = []

    # Si rimuovono spazi iniziali e finali
    text = [t.strip(" ") for t in text if t.strip(" ")]

    # Per ogni paragrafo nel testo
    for i in text:
        # Il carattere di a capo è sostituito con uno spazio vuoto
        i = i.replace('\n', ' ')

        # Si esegue il controllo sulla presenza di stringhe chapter seguite da un numero poiché alcuni paragrafi sono composti unicamente da chapter
        # inseriamo anche i numeri romani
        i = re.sub(r'\bchapter\s+(?:\d+|[ivxlcdm]+)\b', '', i, flags=re.IGNORECASE | re.MULTILINE)

        # Ogni istanza di sequenze di spazi bianchi consecutivi (anche misti: spazi, tab, newline...) è sostituita con un singolo spazio.
        i = re.sub(r'\s+', ' ', i).strip()
        
        if i:
            paragraphs_clean.append(i)

    return paragraphs_clean


def check_length_parag(text):

    """
    La funzione tokenizza il testo, ne conta la lunghezza e mantiene solamente i testi con una lunghezza compresa tra
    50 e 100.

    :args text: testo da analizzare.

    :returns filtered_paragraphs: lista di paragrafi filtrati sulla base della lunghezza.
    """

    # Inizializza una lista di vuota dove saranno raccolti i paragrafi filtrati per lunghezza
    filtered_paragraphs = []

    for i in text:
        # ogni paragrafo nel testo è tokenizzato e ne è controllata la lunghezza
        tokens = nltk.tokenize.word_tokenize(i)
        token_count = len(tokens)

        # se la lunghezza si trova tra questi estremi il paragrafo è mantenuto
        if 50 <= token_count <= 100:
            filtered_paragraphs.append(i)

    return filtered_paragraphs


# Two function to unite the others
def reading(file_path, root):

    """
    La funzione chiama altre due funzioni, abs_path e read_file, restituendone il contenuto.

    :args file_path: il percorso relativo del file;
    :args root: il percorso relativo della cartella che contiene il file.

    :returns text: il contenuto del file in formato stringa.
    """
    # Invoca abs_path per ricostruire il path assoluto del file
    data = abs_path(file_path, root)

    # Invoa read_file per leggere il file e lo ritorna
    text = read_file(data)
    
    return text

def preprocessing(start_match, end_match, data, name):

    """
    La funzione realizza il preprocessing dei dati chiamando le altre funzioni disponibili. Inizia trovando tutte
    le occorrenze di uno ritorno a capo (\n) seguito da un numero variabile di ritorni a capo per simboleggiare la 
    distinzione tra paragrafi e stampa il numero di quelli trovati pre pulizia. Procede poi a chiamare le funzioni
    strip_to_text, paragraphs_sep, cleaning e check_length_parag, ed infine stampa il numero totale di paragrafi rimasti
    post preprocessing.

    :args start_match: stringa iniziale da trovare nel testo, tutto quello che la precede è eliminato
    :args end_match: stringa finale da trovare nel testo, tutto quello che la segue è eliminato
    :args data: il testo su cui effettuare il preprocessing
    :args name: nome del testo

    :returns filtered_parags: lista di paragrafi puliti
    """
    # Si cercano nel testo tutte le occorrenze di due righe vuote consecutive (o separate da spazi bianchi o tab)
    # così da poter controllare la lunghezza del testo in paragrafi prima della pulizia
    matches = re.findall(r'\n\s*\n', data)
    print(f"Numero di paragrafi trovati pre pulizia per {name}: {len(matches)}")
    
    # Invoca strip_to_text
    main_text = strip_to_text(start_match, end_match, data)

    # Invoca paragraphs_sep per separare i paragrafi
    parags = paragraphs_sep(main_text)

    # Invoca la funzione di pulizia dei paragrafi
    parags = cleaning(parags)

    # Invoca la funzione di controllo della lunghezza di ogni paragrafo
    filtered_parags = check_length_parag(parags)

    print(f"Il numero di paragrafi utili di {name} è di: {len(filtered_parags)}")

    return filtered_parags


def save_txt(dict):

    """
    Funzione atta a creare tre cartelle a partire dal dizionario, nelle quali sarà salvato
    ogni paragrafo presente come file .txt separato. 

    :args dict: il dizionario su cui svolgere le analisi

    """

    try:
        # Prende ogni id nel dizionario
        for id in dict:

            # seleziona la path sulla base dell'id
            if 'train' in id:
                path = 'train_UD'
            elif 'val' in id:
                path = 'val_UD'
            elif 'test' in id:
                path = 'test_UD'

            # Crea la cartellal di riferimento se la path non esiste già
            if not os.path.exists(path):
                os.makedirs(path)
                logger.info(f'Cartella {path} creata correttamente.')

            else:
                logger.info(f'Cartella {path} già presente')
        
            # Crea la path specifica del file unendola a quella della cartella
            file = f'{id}.txt'
            file_path = os.path.join(path, file)

            # Seleziona la frase nel dizionario
            sentence = dict[id]['text']

            # Scrive la frase in un file txt unico
            with open(file_path, 'w', encoding='UTF-8') as f:
                f.write(sentence)

    except Exception as e:
        logger.error('Errore nella creazione del file')