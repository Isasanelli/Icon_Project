import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(data):
    # Visualizzazione delle distribuzioni delle variabili
    plot_distributions(data)

def plot_distributions(data):
    cols_to_plot = ['Age', 'Gender', 'Ethnicity', 'SocioeconomicStatus', 'BMI', 'FamilyHistoryDiabetes', 'HbA1c', 'FastingBloodSugar', 'Diagnosis']
    data = data[cols_to_plot]

    plt.figure(figsize=(20, 10))
    for i, col in enumerate(cols_to_plot):
        plt.subplot(3, 3, i + 1)
        plt.hist(data[col], bins=20, edgecolor='k')
        plt.title(col)
        plt.ylabel('Frequenza')
        plt.xlabel('Valore')
    plt.tight_layout()
    plt.suptitle('Distribuzioni delle variabili', y=1.02)
    plt.show()
