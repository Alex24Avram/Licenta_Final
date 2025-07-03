import os
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch.utils.data import DataLoader
from model import get_model

processed_dir = "E:\\Anul 4\\Dataset\\work"
dataset_path = os.path.join(processed_dir, "dataset_splits.pkl")

if os.path.exists(dataset_path):
    with open(dataset_path, "rb") as f:
        train_dataset, test_dataset, val_dataset = pickle.load(f)
    print(f"Seturi de date incarcate!")
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}, Val size: {len(val_dataset)}")
else:
    raise FileNotFoundError("Fisierul dataset_splits.pkl nu exista!")

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def train_model(train_loader, test_loader, val_loader, speaker_folders, model_type="resnet", num_epochs=50, learning_rate=0.0001, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_speakers = len(speaker_folders)

    model = get_model(model_type, num_speakers).to(device)

    model_path = f"trained_model_{model_type}.pth"

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model pre-antrenat incarcat din {model_path}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    epochs_list = []

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    best_epoch = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for i, (mel_spec, labels) in enumerate(train_loader):
            mel_spec, labels = mel_spec.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(mel_spec)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0:
                print(f"Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        epochs_list.append(epoch + 1)

        train_accuracy, _, _ = evaluate_model(model, train_loader, device)
        train_accuracies.append(train_accuracy)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for mel_spec, labels in val_loader:
                mel_spec, labels = mel_spec.to(device), labels.to(device)
                outputs = model(mel_spec)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        val_accuracy, _, _ = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping activat dupa {epoch+1} epoci.")
            print(f"Cel mai mic loss pe validare a fost {best_val_loss:.4f} la epoca {best_epoch}.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("Antrenare finalizata!")
    torch.save(model.state_dict(), model_path)
    print(f"Modelul a fost salvat la {model_path}")

    test_accuracy, y_true, y_pred = evaluate_model(model, test_loader, device)
    print(f"Acuratetea pe setul de test: {test_accuracy:.2f}%")

    plt.figure()
    plt.plot(epochs_list, train_losses, label='Train Loss')
    plt.plot(epochs_list, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Evoluția erorii - Model: {model_type}')
    plt.show()

    plt.figure()
    plt.plot(epochs_list, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_list, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title(f'Evoluția acurateței - Model: {model_type}')
    plt.show()


    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=speaker_folders)
    disp.plot(cmap='Blues')
    plt.title(f'Matrice de Confuzie - {model_type}')
    plt.show()  

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for mel_spec, labels in data_loader:
            mel_spec, labels = mel_spec.to(device), labels.to(device)
            outputs = model(mel_spec)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    return accuracy, y_true, y_pred


if __name__ == "__main__":
    speaker_folders = ["Benjamin_Netanyau", "Jens_Stoltenberg", "Julia_Gillard", "Magaret_Tarcher", "Nelson_Mandela"]

    for model_type in ["baseline", "simple", "resnet"]:
        print(f"\n=== Antrenare model: {model_type.upper()} ===\n")
        train_model(train_loader, test_loader, val_loader, speaker_folders, model_type=model_type)



