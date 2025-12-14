# train.py

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from trump_dataset import TrumpDataset
from trump_model import TrumpNet


# ---------------------------------------------------
# 1. Load & preprocess data
# ---------------------------------------------------

def load_and_prepare_data(csv_path: str):
    data = pd.read_csv(csv_path)

    # Card and column definitions
    cards = [
        'DA','DK','DQ','DJ','D10','D9','D8','D7','D6',
        'HA','HK','HQ','HJ','H10','H9','H8','H7','H6',
        'SA','SK','SQ','SJ','S10','S9','S8','S7','S6',
        'CA','CK','CQ','CJ','C10','C9','C8','C7','C6'
    ]
    forehand = ['FH']
    user = ['user']
    trump = ['trump']

    # Assign new column names
    data.columns = cards + forehand + user + trump

    # Remove user column
    data = data.drop(columns=["user"])

    # Type conversions
    data.trump = data.trump.astype("category")
    data[cards + forehand] = data[cards + forehand].astype(bool)

    # Rename categories
    data.trump = data.trump.cat.rename_categories({
        0: 'DIAMONDS',
        1: 'HEARTS',
        2: 'SPADES',
        3: 'CLUBS',
        4: 'OBE_ABE',
        5: 'UNE_UFE',
        6: 'PUSH',
        10:'PUSH'
    })

    # X = features, y = class indices
    X = data[cards + forehand].astype(int).values
    y = data.trump.cat.codes.values

    # Save mapping for inference
    label_mapping = dict(enumerate(data.trump.cat.categories))

    return X, y, label_mapping


# ---------------------------------------------------
# 2. Train / Val / Test split
# ---------------------------------------------------

def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------
# 3. Training loop
# ---------------------------------------------------

def train_model(model, train_loader, val_loader, device, epochs=20):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds = model.predict(Xb)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_acc = correct / total

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f} - Val Acc: {val_acc:.4f}")

    return model


# ---------------------------------------------------
# 4. Final Test Evaluation
# ---------------------------------------------------

def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model.predict(Xb)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    test_acc = correct / total
    print(f"\nTest Accuracy: {test_acc:.4f}")


# ---------------------------------------------------
# 5. Main – Combine Everything
# ---------------------------------------------------

def main():

    X, y, label_mapping = load_and_prepare_data("2018_10_18_trump (2).csv")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Create datasets
    train_ds = TrumpDataset(X_train, y_train)
    val_ds   = TrumpDataset(X_val, y_val)
    test_ds  = TrumpDataset(X_test, y_test)

    # Create loaders
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False)

    # Create model
    input_dim = X_train.shape[1]     # should be 37
    num_classes = len(set(y))        # should be 7
    model = TrumpNet(input_dim, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train
    model = train_model(model, train_loader, val_loader, device, epochs=20)

    # Evaluate
    evaluate(model, test_loader, device)

    # Save model + label mapping
    torch.save(model.state_dict(), "trump_model.pt")
    torch.save(label_mapping, "trump_labels.pt")

    print("\nModel and label mapping saved.")


if __name__ == "__main__":
    main()
