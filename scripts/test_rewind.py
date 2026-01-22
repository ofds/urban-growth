"""Test rewind functionality."""

import pickle
import logging
import sys
from pathlib import Path

# Add parent directory to path so we can import src modules
sys.path.append('..')

from src.inverse.rewind import RewindEngine

logging.basicConfig(level=logging.INFO)

# Load final city state
with open('../data/processed/sjc.pkl', 'rb') as f:
    final_state = pickle.load(f)

print(f"Final state: {final_state.summary()}")

# Rewind to initial seed
engine = RewindEngine(
    target_ratio=0.25,  # Rewind to 25% of blocks
    removal_strategy='composite'
)

sequence = engine.rewind(final_state)

print(f"\n✓ Generated sequence:")
print(f"  - Initial state: {len(sequence.initial_state.blocks)} blocks")
print(f"  - Final state: {len(sequence.final_state.blocks)} blocks")
print(f"  - Total steps: {len(sequence)} states")

# Save sequence for training
output_path = Path('../data/processed/sjc_sequence.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(sequence, f)

print(f"\n✓ Saved sequence to {output_path}")
