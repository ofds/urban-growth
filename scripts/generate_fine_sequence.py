"""
Generate fine-grained sequence with snapshot_interval=1.
This creates better training data for ML models.
"""

import pickle
import logging
import sys
from pathlib import Path

sys.path.append('..')

from src.inverse.rewind import RewindEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("=" * 60)
    print("Generating Fine-Grained Growth Sequence")
    print("=" * 60)
    
    # Load final city state
    input_path = Path('../data/processed/sjc.pkl')
    print(f"\n📂 Loading final state from {input_path}...")
    
    with open(input_path, 'rb') as f:
        final_state = pickle.load(f)
    
    print(f"   Final state: {len(final_state.blocks)} blocks")
    
    # Rewind with FINE-GRAINED snapshots
    print(f"\n🔄 Running rewind with snapshot_interval=1...")
    print(f"   ⚠️  This will take ~10-15 minutes (creates ~5000 states)")
    
    engine = RewindEngine(
        target_ratio=0.25,
        removal_strategy='composite',
        batch_size=1,              # ONE block per step
        snapshot_interval=1        # EVERY step saved
    )
    
    sequence = engine.rewind(final_state)
    
    print(f"\n✓ Generated fine-grained sequence:")
    print(f"   - States: {len(sequence.states)}")
    print(f"   - Initial blocks: {len(sequence.initial_state.blocks)}")
    print(f"   - Final blocks: {len(sequence.final_state.blocks)}")
    print(f"   - Growth steps: {len(sequence.states) - 1}")
    
    # Save with different name
    output_path = Path('../data/processed/sjc_sequence_fine.pkl')
    print(f"\n💾 Saving to {output_path}...")
    
    with open(output_path, 'wb') as f:
        pickle.dump(sequence, f)
    
    print(f"✓ Saved!")
    print(f"\n📊 File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"\n✅ Done! Use this sequence in train_model.py for best results.")

if __name__ == '__main__':
    main()
