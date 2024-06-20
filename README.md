 # Diabetes Health Dataset Analysis
 ### Introduzione

Questo progetto si concentra sull'analisi di un dataset di salute relativo al diabete, disponibile su Kaggle:
<< https://www.kaggle.com/datasets/rabieelkharoua/diabetes-health-dataset-analysis/data >>. 
L'obiettivo è sviluppare e valutare diversi modelli di apprendimento automatico per predire la diagnosi del diabete basandosi su vari fattori di salute e comportamentali.
Dataset

Il dataset contiene informazioni su vari aspetti della salute e dello stile di vita dei pazienti, tra cui:

    Età (Age)
    Genere (Gender)
    Etnia (Ethnicity)
    Stato socioeconomico (SocioeconomicStatus)
    Livello di istruzione (EducationLevel)
    Indice di massa corporea (BMI)
    Abitudini al fumo (Smoking)
    Consumo di alcol (AlcoholConsumption)
    Attività fisica (PhysicalActivity)
    Glicemia a digiuno (FastingBloodSugar)
    Livelli di HbA1c (HbA1c)
    Colesterolo totale (CholesterolTotal)
    Storia familiare di diabete (FamilyHistoryDiabetes)
    Sindrome dell'ovaio policistico (PolycysticOvarySyndrome)
    Precedente prediabete (PreviousPreDiabetes)
    Ipertensione (Hypertension)

La colonna Diagnosis indica se il paziente è stato diagnosticato con il diabete.
 
## Codice
Nella cartella "src > Code .." troviamo:
- Il modulo  _"**data_loader.py**"_ si occupa di caricare i dati dal file CSV. Utilizza la libreria pandas per leggere il file e gestisce eventuali errori come file non trovato o file vuoto.
- Il modulo _"**feature_engineering.py**" _esegue l'ingegnerizzazione delle caratteristiche per migliorare la qualità dei dati per i modelli di apprendimento automatico. Ad esempio, crea nuovi rapporti e indici basati su variabili esistenti.
- Il modulo _"**model_training.py**"_ è responsabile della preparazione dei dati, dell'addestramento e della valutazione di vari modelli di apprendimento automatico, inclusi K-Nearest Neighbors, Naive Bayes, Regressione Logistica, Decision Tree e Support Vector Machine. Inoltre, addestra un modello di rete neurale MLP (Multi-layer Perceptron).
- Il modulo _"**visualization.py**"_ include funzioni per visualizzare le distribuzioni dei dati e le prestazioni dei modelli. Utilizza librerie come Matplotlib e Seaborn per creare grafici.
- 
## Modelli utilizzati: 

### K-Nearest Neighbors (KNN)

Il modello KNN è semplice e intuitivo, basato sulla vicinanza dei punti dati nel set di addestramento. È utile per piccoli dataset e quando si desidera un modello facilmente interpretabile.

### Naive Bayes

Il modello Naive Bayes è basato sul teorema di Bayes ed è adatto per problemi di classificazione con alte dimensioni. È efficace e rapido, soprattutto con dati categoriali.

### Regressione Logistica

La regressione logistica è un modello statistico che è utile per predire la probabilità di una variabile binaria. È interpretabile e spesso utilizzato come base per modelli più complessi.

### Decision Tree

Il modello ad albero decisionale è facilmente interpretabile e visualizzabile. È adatto per problemi di classificazione e regressione, e funziona bene con dataset che contengono sia variabili numeriche che categoriali.

### Support Vector Machine (SVM)

Il modello SVM è potente per problemi di classificazione e può separare i dati anche in spazi dimensionali elevati. Utilizza una funzione di kernel per trasformare i dati e trovare un confine ottimale tra le classi.

### Multi-layer Perceptron (MLP)

L'MLP è una rete neurale artificiale che può apprendere relazioni non lineari nei dati. È particolarmente utile per problemi complessi dove modelli più semplici non riescono a catturare la struttura dei dati.

## Conclusione

Questo progetto mostra un approccio completo all'analisi e alla modellazione dei dati relativi al diabete, utilizzando diverse tecniche di ingegneria delle caratteristiche, modelli di apprendimento automatico e visualizzazione dei dati. Ogni modello ha i suoi punti di forza e debolezza, e la loro combinazione può fornire una comprensione più approfondita del problema del diabete.
