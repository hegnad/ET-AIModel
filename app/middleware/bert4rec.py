import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict
from torch.utils.data import DataLoader

class BERT4Rec(nn.Module):
    """
    A simple implementation of the BERT4Rec architecture for sequence-based recommendations.
    """
    def __init__(self, num_items: int, embedding_dim: int = 128, num_layers: int = 2, num_heads: int = 2):
        super(BERT4Rec, self).__init__()
        self.embedding = nn.Embedding(num_items, embedding_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads),
            num_layers=num_layers,
        )
        self.output_layer = nn.Linear(embedding_dim, num_items)

    def forward(self, sequence):
        """
        Forward pass through the model.
        """
        embeddings = self.embedding(sequence)
        encoded_sequence = self.encoder(embeddings)
        predictions = self.output_layer(encoded_sequence)
        return predictions


def train_bert4rec(
    model: BERT4Rec,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
):
    """
    Train the BERT4Rec model.

    Args:
        model: BERT4Rec instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data (optional).
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Log training progress
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        # Optional: Validation loss
        if val_loader:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")


def generate_recommendations(model: BERT4Rec, sequence: List[int], top_k: int = 10) -> List[int]:
    """
    Generate recommendations based on the input sequence.

    Args:
        model: Trained BERT4Rec instance.
        sequence: Input sequence of movie IDs.
        top_k: Number of recommendations to generate.

    Returns:
        List of recommended movie IDs.
    """
    model.eval()
    with torch.no_grad():
        sequence_tensor = torch.tensor(sequence).unsqueeze(0)  # Add batch dimension
        predictions = model(sequence_tensor).squeeze(0)
        top_items = torch.topk(predictions[-1], top_k).indices.tolist()
    return top_items


def train_model(
    num_items: int,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    embedding_dim: int = 128,
    num_layers: int = 2,
    num_heads: int = 2,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
):
    """
    Wrapper to initialize and train the BERT4Rec model.
    """
    model = BERT4Rec(num_items, embedding_dim, num_layers, num_heads)
    train_bert4rec(model, train_loader, val_loader, num_epochs, learning_rate)
    torch.save(model.state_dict(), "data/bert4rec_model.pth")
    print("Model trained and saved successfully.")
