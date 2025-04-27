import json
import re
import collections

class WordPieceTokenizer:
    def __init__(self):
        self.vocab = {}
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"

    def preprocess_data(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
        text = text.strip()
        return text

    def split_into_characters(self, word):
        return [word[0]] + [f"##{ch}" for ch in word[1:]]

    def construct_vocabulary(self, corpus, group_no, vocab_size=25):
        word_freq = collections.Counter()

        # Preprocess and count word frequency
        for sentence in corpus:
            processed_sentence = self.preprocess_data(sentence)
            words = processed_sentence.split()
            word_freq.update(words)

        # Initialize vocabulary with character-based splits
        subword_splits = {word: self.split_into_characters(word) for word in word_freq.keys()}
        subwords = collections.defaultdict(int)

        for word, freq in word_freq.items():
            for token in subword_splits[word]:
                subwords[token] += freq

        while len(subwords) < vocab_size:
            pairs = collections.defaultdict(int)
            pair_scores = {}

            # Count occurrences of consecutive subword pairs
            for word, freq in word_freq.items():
                tokens = subword_splits[word]
                for i in range(len(tokens) - 1):
                    pairs[(tokens[i], tokens[i + 1])] += freq  

            if not pairs:
                break

            # Compute the score only for consecutive pairs
            for pair, freq in pairs.items():
                first, second = pair
                if first in subwords and second in subwords:
                    pair_scores[pair] = freq / (subwords[first] * subwords[second])

            # Find the highest scoring consecutive pair
            best_pair = max(pair_scores, key=pair_scores.get)

            # Determine correct merge formatting
            first, second = best_pair
            if first.startswith("##") or second.startswith("##"):
                merged_token = f"{first}{second.replace('##', '')}"
            else:
                merged_token = f"{first}{second}"  # No prefix if the first part is unprefixed

            # Add the merged token to the vocabulary
            subwords[merged_token] = pairs[best_pair]

            # Update the splits by merging the pair in existing words
            for word in list(subword_splits.keys()):
                new_tokens = []
                tokens = subword_splits[word]
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                        new_tokens.append(merged_token)
                        i += 2  # Skip next token as it's merged
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                subword_splits[word] = new_tokens

        # Save the vocabulary
        self.vocab = {word: i for i, word in enumerate(subwords.keys())}
        vocab_filename = f'vocabulary_{group_no}.txt'
        with open(vocab_filename, 'w') as f:
            for token in self.vocab.keys():
                f.write(token + '\n')

        print(f"Vocabulary saved to {vocab_filename}")

    def tokenize(self, sentence):
        """
        Tokenizes a given sentence using the saved vocabulary for the corpus.
        """
        sentence = self.preprocess_data(sentence)
        words = sentence.split()
        tokens = []

        for word in words:
            subword_tokens = []
            i = 0

            while i < len(word):
                found_match = False

                # Try to find the longest subword match from the beginning
                for j in range(len(word), i, -1):
                    subword = word[i:j]
                    if subword in self.vocab:
                        subword_tokens.append(subword)
                        i = j
                        found_match = True
                        break

                # If no match is found, try adding ##
                if not found_match:
                    for j in range(len(word), i, -1):
                        subword = f"##{word[i:j]}"
                        if subword in self.vocab:
                            subword_tokens.append(subword)
                            i = j
                            found_match = True
                            break

                # If still no match, replace with [UNK] and break
                if not found_match:
                    subword_tokens.append(self.unk_token)
                    break

            tokens.extend(subword_tokens)

        return tokens

# Read Corpus from corpus.txt
def read_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.readlines()

# Read Test Data from sample_test.json
def read_test_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Tokenize Test Data
def tokenize_test_file(test_data, tokenizer):
    tokenized_dict = {}
    for item in test_data:
        tokenized_dict[item["id"]] = tokenizer.tokenize(item["sentence"])
    return tokenized_dict

import json

def save_tokenized_sentences(tokenized_sentences, output_file="tokenized_18.json"):
    """
    Saves the tokenized sentences dictionary to a JSON file.
    """
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(tokenized_sentences, f, indent=4)
        print(f"Tokenized sentences saved successfully to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")


# Main Execution
if __name__ == "__main__":
    corpus_file = "corpus.txt"  # File containing the corpus
    # test_file = "sample_test.json"  # File containing id-sentence pairs
    # output_file = "tokenized_18.json"

    # Read corpus and construct vocabulary
    print("################# Reading the Corpus #################")
    corpus = read_corpus(corpus_file)
    print("################# Creating the Vocab from the Corpus #################")
    tokenizer = WordPieceTokenizer()
    vocab = tokenizer.construct_vocabulary(corpus, group_no=18, vocab_size=50000)

    # # Read test data and tokenize
    # print("################# Reading the test data from sample_test_json #################")
    # test_data = read_test_data(test_file)
    # tokenized_sentences = tokenize_test_file(test_data, tokenizer)
    # save_tokenized_sentences(tokenized_sentences)
    # # Print tokenized output
    # print(json.dumps(tokenized_sentences, indent=4))