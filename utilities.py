import torch
import os
import h5py
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

##+++++++++ Parameters And Paths ++++++++++++##
img_dir = "/home/aman/IIITD/semester_4/Deep_Learning /Assignment_3/Code/102flowers/jpg"
txt_dir = "/home/aman/IIITD/semester_4/Deep_Learning /Assignment_3/Code/cvpr2016_flowers/text_c10"

##+++++++++ MetaData ++++++++++++++++++++++++##

def MetaData(txt_dir, img_dir):
    # Get class text paths (excluding .t7 files)
    class_txts = sorted([os.path.join(txt_dir, i) for i in os.listdir(txt_dir) if not i.endswith(".t7")])

    # Dictionary to store image text file paths per class
    class_img_txt_paths = {
        os.path.basename(class_txt): [
            os.path.join(class_txt, j) for j in os.listdir(class_txt) if not j.endswith(".h5")
        ]
        for class_txt in class_txts
    }
    # Dictionary to store image paths per class
    class_img_paths = {
        key: [os.path.join(img_dir, os.path.basename(v).split('.')[0]) for v in value]
        for key, value in class_img_txt_paths.items()
    }

    # List to store (image_path, sentence) tuples
    train_image_text_pairs = []
    test_image_text_pairs = []
    for class_label, txt_paths in class_img_txt_paths.items():
        class_index = int(class_label.split('_')[-1])
        for txt_file in txt_paths:
            image_name = os.path.basename(txt_file).split('.')[0]  # Extract image name
            image_path = os.path.join(img_dir, image_name) + ".jpg"  # Construct image path

            # # Read text file and split into sentences
            with open(txt_file, 'r', encoding='utf-8') as f:
                sentences = f.read().strip().split('\n')  # Splitting based on new line

            pairs = [(image_path,sentence.strip()) for sentence in sentences if sentence.strip()]
            
            if 1<=class_index<=20:
                test_image_text_pairs.extend(pairs)
            else:
                train_image_text_pairs.extend(pairs)

    return train_image_text_pairs,test_image_text_pairs


train_image_text_pairs,test_image_text_pairs = MetaData(txt_dir, img_dir)
# Print sample results
print(f"Train samples: {len(train_image_text_pairs)}")
print(f"Test samples: {len(test_image_text_pairs)}")
print(f"Example Train Sample: {train_image_text_pairs[0]}")
print(f"Example Test Sample: {test_image_text_pairs[0]}")
input("wait")
##++++++ Text to Embeddings +++++++++++++++++##
#################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np

# ✅ Custom WordPiece Tokenizer
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

#  Load Vocabulary
def load_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, "r") as f:
        for i, token in enumerate(f.readlines()):
            vocab[token.strip()] = i
    return vocab

#  Load Trained Word2Vec Model
class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)  #  Include the linear layer

    def forward(self, context):
        embeds = self.embeddings(context)
        out = torch.mean(embeds, dim=1)  # Average word embeddings
        out = self.linear(out)  #  Pass through the linear layer
        return out

#  Load Model Function
def load_model(checkpoint_path, vocab_size, embedding_dim):
    model = Word2VecModel(vocab_size, embedding_dim)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))  # Load model
    model.eval()  # Set to evaluation mode
    return model

#  Convert Text to Embeddings
MAX_TOKENS = 20  # Fixed length for each sentence
EMBEDDING_DIM = 64 
def text_to_embedding(sentence, tokenizer, vocab, model):
    """
    Convert a sentence into a fixed-size tensor (20 words x 64-dim embeddings).
    If a sentence has fewer than 20 words, pad with zeros.
    If a sentence has more than 20 words, truncate it.
    """
    tokens = tokenizer.tokenize(sentence)  # Tokenize sentence
    tokens = tokens[:MAX_TOKENS]  # Truncate if longer than 20 words

    embedding_list = np.zeros((MAX_TOKENS,EMBEDDING_DIM),dtype=np.float32)
    for i,token in enumerate(tokens):
        token_idx = vocab.get(token, vocab.get("[UNK]", 0))  # Default to [UNK] if not in vocab
        token_tensor = torch.tensor([token_idx])  # Convert token to tensor
        
        with torch.no_grad():  # No gradients needed
            embedding = model.embeddings(token_tensor).squeeze().numpy()  # Get embedding
        
        embedding_list[i] = embedding


    return torch.from_numpy(embedding_list)  # (20, 64)


#  Load Vocabulary & Model
vocab_path = "vocabulary_18.txt"
model_path = "word2vec_cbow.pth"

vocab = load_vocab(vocab_path)
word2vec_model = load_model(model_path, vocab_size=len(vocab), embedding_dim=64)
tokenizer = WordPieceTokenizer(vocab)

print(f"Loaded vocabulary with {len(vocab)} words.")
print("Word2Vec Model Loaded Successfully.")

# #  Iterate Over image_text_pairs and Convert Sentences to Embeddings
# image_text_pairs = [
#     ("/path/to/image_1.jpg", "this flower is red with green edges."),
#     ("/path/to/image_2.jpg", "the petals are bright yellow with a brown center."),
# ]

# processed_image_text_pairs = [
#     (image_path, text_to_embedding(text, tokenizer, vocab, word2vec_model))
#     for image_path, text in image_text_pairs
# ]

# #  Print Example
# print(f"Example Processed Sample:\n{processed_image_text_pairs[0]}")

# #  Check Shape
# print(f"Shape of text embedding: {processed_image_text_pairs[0][1].shape}")
# input("wait")
# Assuming `image_text_pairs` contains (image_path, text) tuples
def save_embeddings(image_text_pairs, file_path, model, tokenizer, vocab):
    """
    Converts image text pair into embeddings and save them to disk.
    If the file already exists,it skips computation.
    """
    if os.path.exists(file_path):
        print(f"Skipping embedding generation. File {file_path} already exists.")
        return  # Skip computation if file already exists

    data_embeddings = []
    for image_path, text in image_text_pairs:
        text_embedding = text_to_embedding(text, tokenizer, vocab, model)
        data_embeddings.append((image_path, text_embedding))
    torch.save(data_embeddings, file_path)
    print(f"Embeddings saved to {file_path} successfully!")

# Save train and test embeddings
save_embeddings(train_image_text_pairs, "train_embeddings.pt", word2vec_model, tokenizer, vocab)
save_embeddings(test_image_text_pairs, "test_embeddings.pt", word2vec_model, tokenizer, vocab)

##++++++ Dataset and DataLoader +++++++++++++##


class FlowersDataset(Dataset):
    def __init__(self, embedding_file, transform=None):
        """
        Args:
            image_text_pairs (list of tuples): List containing (image_path, text) pairs.
            transform (callable, optional): Optional transform to be applied on the images.
        """
        self.data = torch.load(embedding_file)
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((32, 32)),  # Resize to 128x128 (optional, adjust as needed)
            transforms.ToTensor(),          # Convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, text_embedding = self.data[idx]

        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")  # Ensure 3-channel image
        image = self.transform(image)

        return image, text_embedding  # Image tensor and raw text



# Create dataset
train_dataset = FlowersDataset("train_embeddings.pt")
test_dataset = FlowersDataset("test_embeddings.pt")



# def collate_fn(batch):
#     """Custom collate function for batching image tensors and padded text embeddings."""
#     images, texts = zip(*batch)  # Unzip the batch into separate lists

#     # Images are already tensors, so just stack them
#     image_tensors = torch.stack(images)

#     # Get max length of text embeddings in the batch
#     max_len = max(text.size(0) for text in texts)

#     # Pad all embeddings to the max length
#     padded_texts = [F.pad(text, (0, max_len - text.size(0)), "constant", 0) for text in texts]

#     return image_tensors, torch.stack(padded_texts)  # Return batch tensors


# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=16, shuffle=False)

# Test data loading
for img, text in test_loader:
    print(f"Image shape: {img.shape}")  # Should be [batch_size, 3, 224, 224]
    print(f"Text sample: {text.shape}")   # Print a few text samples
    break  # Only check the first batch
