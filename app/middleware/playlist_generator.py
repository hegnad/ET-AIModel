import os
import json
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# Dynamically determine the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Load movie catalog and user data
with open(os.path.join(DATA_DIR, "mock_catalog.json"), "r") as movie_file:
    movies = json.load(movie_file)

with open(os.path.join(DATA_DIR, "users_mock_catalog.json"), "r") as user_file:
    users = json.load(user_file)

# Encode movie titles as numeric IDs
movie_encoder = LabelEncoder()
movie_titles = [movie["title"] for movie in movies]
movie_encoder.fit(movie_titles)

# Create user sequences based on their watched property
user_sequences = [
    [idx for idx in movie_encoder.transform(user["watched"]) if idx < len(movie_encoder.classes_)]
    for user in users
    if user["watched"]
]

# Save movie encoder for later use
with open(os.path.join(DATA_DIR, "movie_encoder.json"), "w") as encoder_file:
    json.dump(movie_encoder.classes_.tolist(), encoder_file)

class MovieSequenceDataset(Dataset):
    """
    Custom dataset for movie sequences.
    Each sequence is split into input (x) and target (y).
    """
    def __init__(self, sequences, sequence_length=5):
        self.sequences = sequences
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        x = torch.tensor(sequence[:-1])
        y = torch.tensor(sequence[1:])
        return x, y

def collate_fn(batch):
    """
    Custom collate function to pad sequences in a batch.

    Args:
        batch: List of (x, y) pairs.

    Returns:
        Tuple of padded inputs (x) and targets (y).
    """
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Pad sequences to the maximum length in the batch
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)

    return padded_inputs, padded_targets

# Prepare DataLoader with padding
dataset = MovieSequenceDataset(user_sequences)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Helper function to decode numeric IDs back to movie titles
def decode_sequence(sequence):
    """
    Decode a sequence of numeric IDs back to movie titles.
    """
    return [movie_encoder.classes_[movie_id] for movie_id in sequence]

# Debugging function to print user sequences
def print_user_sequences():
    """
    Print the first few user sequences for debugging purposes.
    """
    for i, sequence in enumerate(user_sequences[:5]):
        print(f"User {i + 1} Sequence: {decode_sequence(sequence)}")
