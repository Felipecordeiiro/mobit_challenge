import pandas as pd
import scipy
from sklearn.model_selection import train_test_split

def get_data():
    """
    Carrega e prepara os dados do dataset BMW10.
    
    Retorna:
        train_df (pd.DataFrame): DataFrame de treino.
        val_df (pd.DataFrame): DataFrame de validação.
        test_df (pd.DataFrame): DataFrame de teste.
    """
    # Carregar anotações

    mat = scipy.io.loadmat('data/bmw10_release/bmw10_annos.mat')
    annots = mat['annos'][0]

    filepaths, labels = [], []
    for entry in annots:
        img_path = entry[0][0]         
        label = int(entry[1][0][0])    
        filepaths.append(img_path)
        labels.append(label)

    df = pd.DataFrame({'filepath': filepaths, 'label': labels})

    # Filtrar classes 3, 4, 5
    wanted = [3, 4, 5]
    df['label_filtered'] = df['label'].apply(lambda x: x if x in wanted else 0)  # 0 = Outros

    # Mapear para [0: Outros, 1: 3, 2: 4, 3: 5] (opcional, para facilitar a leitura da matriz)
    mapping = {0:0, 3:1, 4:2, 5:3}
    df['class_final'] = df['label_filtered'].map(mapping)

    print("=== DISTRIBUIÇÃO ORIGINAL ===")
    print(df['class_final'].value_counts().sort_index())
    class_counts = df['class_final'].value_counts().sort_index()
    print(f"Classe 0 (Outros): {class_counts[0]} amostras")
    for i in range(1, 4):
        if i in class_counts:
            print(f"Classe {i}: {class_counts[i]} amostras")

    train_df, test_df = train_test_split(df, test_size=0.25, stratify=df['class_final'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df['class_final'], random_state=42)

    return train_df, val_df, test_df