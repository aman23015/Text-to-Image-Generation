import re
import json
import torch
import collections
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize the tokenizer using the loaded vocabulary
class WordPieceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.unk_token = "[UNK]"

    def preprocess_data(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
        text = text.strip()
        return text

    def tokenize(self, sentence):
        sentence = self.preprocess_data(sentence)
        words = sentence.split()
        tokens = []

        for word in words:
            subword_tokens = []
            i = 0
            while i < len(word):
                found_match = False
                for j in range(len(word), i, -1):
                    subword = word[i:j]
                    if subword in self.vocab:
                        subword_tokens.append(subword)
                        i = j
                        found_match = True
                        break
                if not found_match:
                    for j in range(len(word), i, -1):
                        subword = f"##{word[i:j]}"
                        if subword in self.vocab:
                            subword_tokens.append(subword)
                            i = j
                            found_match = True
                            break
                if not found_match:
                    subword_tokens.append(self.unk_token)
                    break

            tokens.extend(subword_tokens)

        return tokens


# Custom Dataset Class(Making the data just to train the word2vec by using CBOW technique)
class Word2VecDataset(Dataset):
    def __init__(self, corpus_file, tokenizer, window_size=2):
        with open(corpus_file, "r") as f:
            self.corpus = f.readlines()
        
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.data = []
        self.word_to_idx = tokenizer.vocab
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        self.prepare_data()

    def prepare_data(self):
        for sentence in self.corpus:
            tokens = self.tokenizer.tokenize(sentence)
            indexed_tokens = [self.word_to_idx[token] for token in tokens if token in self.word_to_idx]

            for i in range(len(indexed_tokens)):
                context = []
                target = indexed_tokens[i]

                # Create context words
                for j in range(-self.window_size, self.window_size + 1):
                    if j != 0 and (0 <= i + j < len(indexed_tokens)):
                        context.append(indexed_tokens[i + j])

                if len(context) == 2 * self.window_size:
                    self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])


# CBOW Model
class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        embeds = self.embeddings(context)
        out = torch.mean(embeds, dim=1)  # Average context word embeddings
        out = self.linear(out)
        return out


# Training Function
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, checkpoint_path):
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for context, target in train_loader:
            context = context.to(device)
            target = target .to(device)
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for context, target in val_loader:
                context = context.to(device)
                target = target .to(device)
                output = model(context)
                loss = criterion(output, target)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save model checkpoint
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")

    # Plot Training and Validation Loss
    output_file = "training_loss_plot.png"
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(output_file,format="png",dpi=300)
    print(f"Training loss plot saved as '{output_file}'")
    plt.show()

# Load Vocabulary
def load_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, "r") as f:
        for i, token in enumerate(f.readlines()):
            vocab[token.strip()] = i
    return vocab

# Load corpus data
def load_corpus(corpus_file):
    with open(corpus_file, "r", encoding="utf-8") as f:
        return f.readlines()
    
# Load trained model
def load_model(checkpoint_path, vocab_size, embedding_dim):
    model = Word2VecModel(vocab_size, embedding_dim)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))  # Safe loading
    model.eval()  # Set to evaluation mode
    return model    

# Function to find two triplets
def find_triplets(cosine_sim_matrix, token_list, num_triplets=2):
    triplets = []

    for i in range(len(token_list)):  # Iterate over all words
        word = token_list[i]
        similarities = cosine_sim_matrix[i].copy()

        # Exclude self-similarity
        similarities[i] = -np.inf

        # Find the most similar word
        most_similar_idx = np.argmax(similarities)
        most_similar_word = token_list[most_similar_idx]

        # Find the most dissimilar word (excluding -np.inf)
        similarities_masked = similarities.copy()
        similarities_masked[similarities_masked == -np.inf] = np.inf  # Convert -inf to inf to ignore it
        most_dissimilar_idx = np.argmin(similarities_masked)
        most_dissimilar_word = token_list[most_dissimilar_idx]

        # Ensure all selected words are different
        if (
            most_similar_word != word and
            most_dissimilar_word != word and
            most_similar_word != most_dissimilar_word
        ):
            # Save triplet
            triplets.append((word, most_similar_word, most_dissimilar_word))

        # Stop when we have enough triplets
        if len(triplets) >= num_triplets:
            break

    return triplets

def tokenize_and_get_unique_tokens(corpus, tokenizer):
    """
    Tokenizes all words in the given corpus and extracts unique tokens.
    """
    tokenized_corpus = []

    for line in corpus:
        tokenized_corpus.extend(tokenizer.tokenize(line))

    # Keep only unique tokens
    unique_tokens = list(set(tokenized_corpus))

    print(f"Total unique tokens after tokenization: {len(unique_tokens)}")
    return unique_tokens

def extract_token_embeddings(unique_tokens, vocab, model):
    """
    Extracts embeddings for each unique token using the trained Word2Vec model.
    """
    token_embeddings = {}

    for token in unique_tokens:
        token_idx = vocab.get(token, len(vocab))  # Assign new index if missing
        token_tensor = torch.tensor([token_idx])
        
        with torch.no_grad():  # Disable gradient computation
            embedding = model.embeddings(token_tensor).squeeze().numpy()
        
        token_embeddings[token] = embedding

    print(f"Stored embeddings for {len(token_embeddings)} unique tokens.")
    return token_embeddings



# Main Execution
if __name__ == "__main__":
    vocab_file = "vocabulary_18.txt"  # Updated vocabulary file
    corpus_file = "corpus.txt"  # Corpus for training
    model_path = "word2vec_cbow.pth"

    # Load Vocabulary and Initialize Tokenizer
    vocab = load_vocab(vocab_file)
    tokenizer = WordPieceTokenizer(vocab)

    # Prepare Dataset
    dataset = Word2VecDataset(corpus_file, tokenizer)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize Model
    print("Length of the vocab : ",len(vocab))
    embedding_dim = 64  # Word vector size
    model = Word2VecModel(vocab_size=len(vocab), embedding_dim=embedding_dim)
    model.to(device)
    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)

    # Train Model
    num_epochs = 50
    checkpoint_path = "word2vec_cbow.pth"
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs, checkpoint_path)

    # # Load vocabulary and corpus
    # vocab = load_vocab(vocab_file)
    # corpus = load_corpus(corpus_file)

    # print(f"Loaded vocabulary size: {len(vocab)}")
    # print(f"Loaded corpus size: {len(corpus)}")

    # unique_tokens = tokenize_and_get_unique_tokens(corpus, tokenizer)

    # model = load_model(model_path, len(vocab), embedding_dim)

    # token_embeddings = extract_token_embeddings(unique_tokens, vocab, model)

    # # Convert embeddings dictionary to matrix
    # embedding_matrix = np.array(list(token_embeddings.values()))
    # token_list = list(token_embeddings.keys())

    # # Compute cosine similarity matrix
    # cosine_sim_matrix = cosine_similarity(embedding_matrix)

    # print("Cosine similarity matrix shape:", cosine_sim_matrix.shape)

    # # Get triplets
    # triplets = find_triplets(cosine_sim_matrix, token_list, num_triplets=10)

    # # Print triplets
    # for i, triplet in enumerate(triplets, 1):
    #     print(f"Triplet {i}: {triplet}")
