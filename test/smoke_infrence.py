"""
Smoke test to verify inference works end-to-end
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import load_bundle, preprocess_input, predict_next_word

def test_inference_basic():
    """Test that the model can load and make predictions"""
    # Load model and artifacts
    print("Loading model bundle...")
    bundle = load_bundle()
    print(f"✓ Model loaded successfully on device: {bundle.device}")
    
    # Test preprocessing
    test_text = "the quick brown fox jumps over the lazy"
    print(f"\nTesting with input: '{test_text}'")
    processed = preprocess_input(
        text=test_text,
        word2idx=bundle.word2idx,
        unknown_idx=bundle.unknown_idx,
        sequence_length=bundle.sequence_length,
    )
    print(f"✓ Preprocessing successful")
    print(f"  - Preprocessed text: '{processed['text']}'")
    print(f"  - Input tensor shape: {processed['input_tensor'].shape}")
    
    # Test prediction
    predicted_word = predict_next_word(bundle, processed["input_tensor"])
    print(f"✓ Prediction successful")
    print(f"  - Predicted next word: '{predicted_word}'")
    
    # Validate output
    assert predicted_word, "Predicted word should not be empty"
    assert isinstance(predicted_word, str), "Predicted word should be a string"
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_inference_basic()
