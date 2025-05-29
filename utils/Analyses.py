from collections import Counter
import matplotlib.pyplot as plt

def analyze_distribution(dataset, dataset_name):
    print(f"\n=== ANÁLISE {dataset_name.upper()} ===")
    labels = [sample[1] for sample in dataset.samples]
    counts = Counter(labels)
    
    total = len(labels)
    for class_id in sorted(counts.keys()):
        count = counts[class_id]
        percentage = (count / total) * 100
        class_name = {0: "Outros", 1: "Classe 3", 2: "Classe 4", 3: "Classe 5"}[class_id]
        print(f"{class_name}: {count} amostras ({percentage:.1f}%)")
    
    return counts

def plot_distribution_comparison(train_df, train_dataset, val_dataset, test_dataset):

    train_counts = analyze_distribution(train_dataset, "TREINO")
    val_counts = analyze_distribution(val_dataset, "VALIDAÇÃO") 
    test_counts = analyze_distribution(test_dataset, "TESTE")

    # Distribuição original do treino
    original_counts = train_df['class_final'].value_counts().sort_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Distribuição original
    class_names = ["Outros", "Classe 3", "Classe 4", "Classe 5"]
    ax1.bar(range(len(original_counts)), original_counts.values)
    ax1.set_title('Distribuição Original (Treino)')
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Número de Amostras')
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45)
    
    # Distribuição balanceada
    balanced_counts = [train_counts[i] for i in sorted(train_counts.keys())]
    ax2.bar(range(len(balanced_counts)), balanced_counts)
    ax2.set_title('Distribuição Balanceada (Treino)')
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Número de Amostras')
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45)
    
    plt.tight_layout()
    plt.show()