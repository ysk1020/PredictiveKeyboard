import torch
import torch.nn as nn 
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
import json
# ------------------------
# Model & tokenizer config
# ------------------------

sequence_length = 4

with open('model/vocab.json', 'r') as f:
    vocab_data = json.load(f)

word2idx = vocab_data["word2idx"]
idx2word = {i: word for word, i, in word2idx.items()}
vocab_size = len(word2idx)


def encode(seq: list[str]) -> list[int]:
    """Convert a list of words to a list of indices."""
    return [word2idx[word] for word in seq]

# ------------------------
# Model definition
# ------------------------

class PredictiveKeyboardModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 128 ):
        super().init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)              # converts word indices into dense vectors.
        output, _ = self.lstm(x)     # captures the sequential context of the input.
        last_step= output[:,-1,:]      
        logits = self.fc(last_step) # feed to get a vector of size "vocab_size", respresenting the predicted probabilities for each word in the vocabulary
        return logits


# ------------------------
# Inference helper
# ------------------------

@torch.inference_mode()
def suggest_next_words(model: PredictiveKeyboardModel, text_prompt: str, top_k: int = 3) -> list[str]:
    model.eval()
    tokens = word_tokenize(text_prompt.lower())

    if len(tokens) < sequence_length:
        raise ValueError(f"Text prompt should have at least {sequence_length-1} words.")

    input_seq = tokens[-(sequence_length-1):]
    input_tensor = torch.tensor(encode(input_seq)).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=-1).squeeze(0)
        top_indices = torch.topk(probabilities, top_k, dim=-1).indices.tolist()

    suggested_words = [idx2word[idx] for idx in top_indices]
    return suggested_words