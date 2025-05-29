import torch

def train_model(model, trainloader, valloader, criterion, optimizer, device='cuda', num_epochs=10):
    model.to(device)
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Validação
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss/len(trainloader):.4f}, "
              f"Val Loss: {val_loss/len(valloader):.4f}, Val Acc: {correct/total:.4f}")
    return model