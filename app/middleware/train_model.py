from bert4rec import train_model
from playlist_generator import data_loader, movie_encoder

# Define paths
MODEL_SAVE_PATH = "data/bert4rec_model.pth"

def main():
    """
    Main function to train the BERT4Rec model.
    """

    # Verify the correct number of items
    num_items = len(movie_encoder.classes_)
    print(f"Number of unique items (movies): {num_items}")

    # Training parameters
    embedding_dim = 128
    num_layers = 2
    num_heads = 2
    num_epochs = 10
    learning_rate = 0.001

    # Train the model
    print("Starting model training...")
    train_model(
        num_items=num_items,
        train_loader=data_loader,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
    )


if __name__ == "__main__":
    main()
