import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch

def compute_class_weights(train_df, device='cuda'):
    """
    Calcula os pesos das classes baseados na frequência inversa.
    
    Args:
        train_df (pd.DataFrame): DataFrame contendo a coluna 'class_final'.
        
    Returns:
        torch.FloatTensor: Tensor com os pesos das classes.
    """
    class_weights = compute_class_weight('balanced', 
                                         classes=np.unique(train_df['class_final']), 
                                         y=train_df['class_final'])
    
    print("Calculated Class Weights:", class_weights)
    
    # Converter para tensor
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print("Class Weights Tensor shape:", class_weights_tensor.shape)
    
    return class_weights_tensor