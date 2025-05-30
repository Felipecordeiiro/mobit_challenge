import torch
from utils.metrics import generating_graphs

def train_model(model, model_name, trainloader, valloader, criterion, optimizer, device='cuda', num_epochs=10):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    model.to(device)
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()

        correct = 0
        total = 0
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_losses.append(running_loss / len(trainloader))
        train_accuracies.append(correct / total)

        # Validação
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_losses.append(val_loss / len(valloader))
        val_accuracies.append(val_correct / val_total)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss/len(trainloader):.4f}, "
              f"Val Loss: {val_loss/len(valloader):.4f}, Val Acc: {correct/total:.4f}")
    
    generating_graphs(num_epochs, model_name, train_losses, val_losses, train_accuracies, val_accuracies)
    
    return model