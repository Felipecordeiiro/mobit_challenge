from collections import Counter
from torch.utils.data import Dataset

class BMWDataset(Dataset):
    def __init__(self, dataframe, img_dir, basic_transform, augmented_transform=None, 
                 minority_classes=[1, 2, 3], augment_factor=3):
        """
        Args:
            dataframe: DataFrame com colunas 'filepath' e 'class_final'
            img_dir: diretório base das imagens
            basic_transform: transform básico para classe majoritária
            augmented_transform: transform com augmentation para classes minoritárias
            minority_classes: lista das classes minoritárias (default: [1, 2, 3])
            augment_factor: quantas vezes aumentar as classes minoritárias
        """
        self.img_dir = img_dir
        self.basic_transform = basic_transform
        self.augmented_transform = augmented_transform or basic_transform
        self.minority_classes = minority_classes
        self.augment_factor = augment_factor
        
        # Separar dados por classe
        self.majority_samples = []
        self.minority_samples = []
        
        for _, row in dataframe.iterrows():
            sample = (row['filepath'], row['class_final'])
            if row['class_final'] in minority_classes:
                self.minority_samples.append(sample)
            else:
                self.majority_samples.append(sample)
        
        # Criar dataset balanceado
        self.samples = []
        
        # Adicionar amostras da classe majoritária (sem repetição)
        for sample in self.majority_samples:
            self.samples.append((*sample, False))  # False = não usar augmentation
        
        # Adicionar amostras das classes minoritárias (com repetição e augmentation)
        for sample in self.minority_samples:
            # Amostra original
            self.samples.append((*sample, False))
            # Amostras aumentadas
            for _ in range(self.augment_factor - 1):
                self.samples.append((*sample, True))  # True = usar augmentation
        
        print(f"=== DATASET BALANCEADO ===")
        print(f"Amostras originais majoritárias: {len(self.majority_samples)}")
        print(f"Amostras originais minoritárias: {len(self.minority_samples)}")
        print(f"Total de amostras após balanceamento: {len(self.samples)}")
        
        # Contar distribuição final
        final_counts = Counter([sample[1] for sample in self.samples])
        for class_id in sorted(final_counts.keys()):
            print(f"Classe {class_id}: {final_counts[class_id]} amostras")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath, label, use_augmentation = self.samples[idx]
        
        # Carregar imagem
        full_path = os.path.join(self.img_dir, filepath)
        image = Image.open(full_path).convert('RGB')
        
        # Aplicar transform apropriado
        if use_augmentation and self.augmented_transform:
            image = self.augmented_transform(image)
        else:
            image = self.basic_transform(image)
        
        return image, label