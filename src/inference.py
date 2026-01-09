
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
import nltk
from nltk.tokenize import word_tokenize

from scripts.predictive_keyboard_model import Config, PredictiveKeyboard

# Paths
artifacts_dir = Path("../artifacts")
WEIGHTS_PATH = artifacts_dir / "weights.pt"
CONFIG_PATH = artifacts_dir / "config.json"
VOCAB_PATH = artifacts_dir / "vocab.json"

@dataclass
class ModelBundle:
    pass

def choose_device():
    pass

def load_config():
    pass

def load_vocab():
    pass

def load_model():
    pass

def load_bundle():
    pass

def preprocess_input():
    pass

def predict_next_word():
    pass

def write_output():
    pass

def main():
    pass