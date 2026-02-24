# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Include the problem statement and Dataset
</br>
</br>
</br>

## DESIGN STEPS
STEP 1:
Import required libraries, load the dataset, and define training & testing datasets.

STEP 2:
Initialize the model, loss function, and optimizer. Use CrossEntropyLoss for multi-class classification and Adam optimizer for efficient training.

STEP 3:
Train the model using the training dataset with forward and backward propagation.

STEP 4:
Evaluate the model on the testing dataset to measure accuracy and performance.

STEP 5:
Make predictions on new data using the trained model

## PROGRAM
Include your code here
```
# Load Pretrained Model and Modify for Transfer Learning

from torchvision.models import VGG19_Weights
model=models.vgg19(weights=VGG19_Weights.DEFAULT)

# Modify the final fully connected layer to match the dataset classes

num_classes = len(train_dataset.classes)
model.classifier[6] = nn.Linear(4096, num_classes)

# Include the Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[-1].parameters(), lr=0.001)

# Train the model

def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')
    # Plot training and validation loss
    print("Name: KUSHMA")
    print("Register Number: 212224040168")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
<img width="996" height="762" alt="image" src="https://github.com/user-attachments/assets/cc040090-932a-4691-b025-ea0edb328aab" />

### Confusion Matrix
<img width="983" height="812" alt="image" src="https://github.com/user-attachments/assets/b41aa5a5-98a0-4932-b963-5f04546419a0" />

### Classification Report
<img width="665" height="455" alt="image" src="https://github.com/user-attachments/assets/37d73f16-66c1-40ac-8687-48c6efe72a31" />

### New Sample Prediction
<img width="466" height="447" alt="image" src="https://github.com/user-attachments/assets/fd259b89-a79d-41cd-977b-7083455a33a7" />

## RESULT
Thus, the Transfer Learning for classification using the VGG-19 architecture has been successfully implemented.
