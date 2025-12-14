import torch
import torch.nn as nn

class TrumpNet(nn.Module):
    """
    Wide and deep MLP for trump prediction.
    Works well with engineered features + one-hot card bits.
    """

    def __init__(self, input_dim=37, num_classes=7):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(torch.softmax(logits, dim=1), dim=1)


class TrumpMLP:
    def __init__(self, model_path="../models/trump_model.pt", device="cpu"):
        self.device = device

        # build architecture (must match training!)
        self.model = TrumpNet().to(device)

        # load saved weights
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_trump(self, hand_cards, forehand_flag):
        """
        hand_cards: list of card strings like ["HA", "S8", ...]
        forehand_flag: 0 or 1
        """

        # Convert card strings to 36-length mask
        from cards import CARD_TO_INDEX  # you'll need this file
        vec = torch.zeros(37)                     # 36 + 1 forehand
        for c in hand_cards:
            vec[CARD_TO_INDEX[c]] = 1

        vec[-1] = forehand_flag  # last index = forehand

        vec = vec.to(self.device).float()

        with torch.no_grad():
            if isinstance(vec, torch.Tensor):
                x = vec.clone().detach().unsqueeze(0).float()
            else:
                x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
            logits = self.model(x)
            trump_index = torch.argmax(logits).item()

        return trump_index
