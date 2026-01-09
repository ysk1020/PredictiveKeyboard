# train and test 
from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import nltk
from nltk.tokenize import word_tokenize

# Config
@dataclass
class Config:
  sequence_length: int = 4
  embed_dim: int = 64
  hidden_dim: int = 128
  lr: float = 0.001
  epochs: int = 20
  max_train_samles: int = 10000 
  top_k: int = 3  

# Utilities

def ensure_nltk_resources() -> None:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("tokenizers/punkt_tab/english")
    except LookupError:
        nltk.download("punkt_tab")

# Creating Vocabulary
"""
1.   a dictionary to map each word to an index
2.   and another dictionary to reverse it back

Here, we counted how often each word appears using Counter, then sorted the vocabulary from most to least frequent. This sorted list helps us assign lower indices to more common words (useful for embeddings). Then, we created word2idx and idx2word dictionaries to convert words to unique IDs and back. Finally, we stored the total vocabulary size, which will define the input and output dimensions for our model.
"""

def build_vocab(tokens: List[str])-> Tuple[dict, dict]:
  counts = Counter(tokens)
  vocab = sorted(counts, key=counts.get, reverse=True)

  if "<unk>" not in vocab:
    vocab = ["<unk>"] + vocab
  

  word2idx = {word: i for i, word in enumerate(vocab)}
  idx2word = {i: word for i, word in enumerate(vocab)}
  return word2idx, idx2word

"""
Input-output sequences

To predict the next word, the model needs context. We can use a sliding window approach. So, let’s create input-target sequences for next word prediction:
"""

def encode(tokens: List[str],word2idx: dict)-> List[int]:
  unk = word2idx.get('<unk>', 0)
  return [word2idx.get(word, word2idx["<unk>"]) for word in tokens]


def make_dataset(tokens:List[str], cfg: Config)-> List[Tuple[torch.Tensor, torch.Tensor]]:
  data = []
  for i in range(len(tokens) - cfg.sequence_length):
    input_seq = tokens[i:i + cfg.sequence_length]
    target_word = tokens[i + cfg.sequence_length]
    data.append((input_seq, target_word))
  return data


"""
1. A tensor is like a massive container of numbers (often multi-dimensional) that holds your data in a structured form;

2. The shape tells the model how that data is organized — for example:

    * how many sequences per batch,
    * how long each sequence is,
  how many features (like embedding dimensions) each element has.
  yes — the shape defines what kind of data structure the layer expects and outputs.

Designing the model architecture
"""

class PredictiveKeyboard(nn.Module):
  def __init__(self, vocab_size:int, embed_dim: int=64, hidden_dim: int=128):
    super(PredictiveKeyboard, self).__init__()
    self.embedding = nn.Embedding(vocab_size,embed_dim)
    self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
    self.fc=nn.Linear(hidden_dim, vocab_size)

  def forward(self,x):
    x = self.embedding(x)              # converts word indices into dense vectors.
    output, _ = self.lstm(x)           # captures the sequential context of the input.
    output = self.fc(output[:, -1, :]) # feed to get a vector of size "vocab_size", respresenting the predicted probabilities for each word in the vocabulary
    return output

# Training the model
def train_model(
    text_path: Path,
    artifacts_dir: Path,
    cfg: Config,
)-> None:
  ensure_nltk_resources()
  text = text_path.read_text(encoding='utf-8', errors='ignore').lower()
  tokens = word_tokenize(text)
  word2idx, idx2word = build_vocab(tokens)
  vocab_size = len(word2idx)
  pairs = make_dataset(tokens, cfg)

  # Encode pairs into tensors
  encoded_pairs = [
    (
      torch.tensor(encode(inp, word2idx), dtype=torch.long),
      torch.tensor(word2idx.get(tgt, word2idx["<unk>"]), dtype=torch.long)
    )
    for inp, tgt in pairs
  ]
  device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = PredictiveKeyboard(vocab_size, cfg.embed_dim, cfg.hidden_dim).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

  for epoch in range(cfg.epochs):
      total_loss = 0.0
      seen = 0
      random.shuffle(encoded_pairs)
      for input_seq, target in encoded_pairs[:cfg.max_train_samles]:
          input_seq = input_seq.unsqueeze(0).to(device)
          target = target.unsqueeze(0).to(device)
          
          output = model(input_seq)
          loss = criterion(output, target)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          total_loss += loss.item()
          seen += 1

      avg = total_loss / max(seen, 1)
      print(f"Epoch {epoch+1}/{cfg.epochs}  loss={avg:.4f}")

  # Save artifacts
  
  artifacts_dir.mkdir(parents=True, exist_ok=True)
  torch.save(model.state_dict(), artifacts_dir / "weights.pt")
  (artifacts_dir / "vocab.json").write_text(
    json.dumps({"word2idx": word2idx, "idx2word": idx2word}, ensure_ascii=False),
    encoding='utf-8'
  )

  (artifacts_dir / "config.json").write_text(
    json.dumps(cfg.__dict__, indent=2),
    encoding='utf-8'
  )

  print(f"Saved: {artifacts_dir/'weights.pt'}")
  print(f"Saved: {artifacts_dir/'vocab.json'}")
  print(f"Saved: {artifacts_dir/'config.json'}")

"""
# Predicting the next words
"""
@torch.inference_mode()
def suggest_next_words(prompt:str, artifacts_dir: Path, top_k: int =3)-> List[str]:
  ensure_nltk_resources()

  cfg = Config(**json.loads((artifacts_dir / "config.json").read_text(encoding='utf-8')))
  vocab_data = json.loads((artifacts_dir / "vocab.json").read_text(encoding='utf-8'))
  word2idx = vocab_data["word2idx"]
  idx2word = {int(k): v for k, v in vocab_data["idx2word"].items()}

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = PredictiveKeyboard(len(word2idx), cfg.embed_dim, cfg.hidden_dim).to(device)
  model.load_state_dict(torch.load(artifacts_dir / "weights.pt", map_location=device))
  model.eval()
  tokens = word_tokenize(prompt.lower())
  if len(tokens) < cfg.sequence_length:
    raise ValueError(f"Text prompt should have at least {cfg.sequence_length-1} words.")

  context = tokens[-cfg.sequence_length:]
  x = torch.tensor(encode(context, word2idx), dtype=torch.long, device=device)
  logits = model(x)
  probs = F.softmax(logits, dim=1).squeeze()
  
  top = torch.topk(probs, k=min(top_k, probs.numel())).indices.tolist()
  print('Suggestions:', suggest_next_words(model, 'So, are we really at'))
  return [idx2word[idx] for idx in top]

def main():
    cfg = Config(epochs=3)  # start small locally
    text_path = Path("../data/sherlock-holmes_stories_plain-text_advs.txt")
    artifacts_dir = Path("../artifacts")

    train_model(text_path=text_path, artifacts_dir=artifacts_dir, cfg=cfg)

    print(suggest_next_words("So, are we really at", artifacts_dir, top_k=5))


if __name__ == "__main__":
    main()