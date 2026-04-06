
'''
Docstring for inference
funcontions:
1. lowercase the input text
2. tokenize the input text
3. ensure enough context tokens exist
4. takes last sequence_length tokens 
5. encodes to IDs using word2idx
6. handles unknown words by mapping to <UNK> or dropping them
'''
import json
from pathlib import Path
from dataclasses import dataclass

import torch
import unicodedata
import nltk
from nltk.tokenize import word_tokenize

from scripts.predictive_keyboard_model import Config, PredictiveKeyboard

# Paths
CONFIG_NAME = "config.json"
VOCAB_NAME = "vocab.json"
WEIGHTS_NAME = "weights.pt"

@dataclass
class ModelBundle:  
    model: PredictiveKeyboard
    word2idx: dict[str, int]
    idx2word: dict[int, str]
    cfg: Config
    device: torch.device
    unknown_idx: int
    sequence_length: int

def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    

def load_config(CONFIG_PATH: Path)-> Config:
    cfg = Config(**json.loads((CONFIG_PATH).read_text(encoding='utf-8')))
    return cfg

def load_vocab(VOCAB_PATH: Path) -> tuple[dict[str, int], dict[int, str], int]:
    vocab_data = json.loads((VOCAB_PATH).read_text())
    word2idx = vocab_data['word2idx']
    idx2word = {int(k): v for k, v in vocab_data["idx2word"].items()}
    unknown_idx = word2idx.get("<unk>")
    if unknown_idx is None:
        unknown_idx = word2idx.get("<UNK>")
    if unknown_idx is None:
        raise KeyError("Neither '<unk>' nor '<UNK>' found inside word2idx")
    return word2idx, idx2word, unknown_idx

def load_model(weights_path: str|Path, vocab_size:int, cfg, device:torch.device)-> PredictiveKeyboard:
    model=PredictiveKeyboard(vocab_size=vocab_size, embed_dim=cfg.embed_dim, hidden_dim=cfg.hidden_dim)
    # load weights 
    state=torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    # move model to device 
    model= model.to(device)
    # enable evaluation mode 
    model.eval()
    return model


def load_bundle():
    base_dir =Path(__file__).resolve().parent
    artifacts_dir=base_dir.parent/'artifacts'

    # --- Paths ---
    cfg = load_config(artifacts_dir / CONFIG_NAME)
    word2idx, idx2word, unknown_idx = load_vocab(artifacts_dir / VOCAB_NAME)

    weights_path = artifacts_dir / WEIGHTS_NAME
    # --- Devices ---
    device = choose_device()

    # --- Load Model ---
    vocab_size=len(word2idx)
    sequence_length=cfg.sequence_length

    model = load_model(weights_path, vocab_size, cfg, device)
    return ModelBundle(model=model, word2idx=word2idx, idx2word=idx2word, cfg=cfg, device=device, unknown_idx=unknown_idx, sequence_length=sequence_length)

def preprocess_input(text:str, word2idx: dict[str, int], unknown_idx: int, sequence_length: int) -> dict:
    text = text.lower()
    text = text.strip()
    text = unicodedata.normalize('NFKD', text)
    tokens = word_tokenize(text)
    truncated = False
    if sequence_length is not None and len(tokens) > sequence_length:
        tokens = tokens[-sequence_length:]
        truncated = True

    # Map tokens -> ids using vocab, fallback to unknown_idx for OOV
    input_ids = [word2idx.get(token, unknown_idx) for token in tokens]

    # Convert to tensor with batch dimension: (1, seq_len)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    return {"text": text, "input_tensor": input_tensor, "truncated": truncated}

def predict_next_word(bundle: ModelBundle, input_tensor: torch.Tensor) -> str:
    model = bundle.model
    device= bundle.device

    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        predicted_idx = torch.argmax(logits, dim=-1).item()
    predicted_word = bundle.idx2word.get(predicted_idx, "<unk>")
    return predicted_word

def write_output(raw_text: str, predicted_word: str, truncated: bool    ) -> None:
    print(f"input: {raw_text}")
    print(f"predicted word: {predicted_word}")
    if truncated:
        print(f"truncated: {truncated}")

def main() -> None:
    bundle = load_bundle()

    raw_text = input("Enter text: ").strip()
    if not raw_text:
        print("No input provided.")
        return

    processed = preprocess_input(
        text=raw_text,
        word2idx=bundle.word2idx,
        unknown_idx=bundle.unknown_idx,
        sequence_length=bundle.sequence_length,
    )

    input_tensor = processed["input_tensor"]

    if input_tensor.numel() == 0:
        print("No valid tokens were produced from the input.")
        return

    predicted_word = predict_next_word(bundle, input_tensor)

    write_output(
        raw_text=processed["text"],
        predicted_word=predicted_word,
        truncated=processed["truncated"],
    )

if __name__ == "__main__":
    main()