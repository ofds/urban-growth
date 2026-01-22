"""
Train urban growth model from rewind sequence.

Usage:
    python scripts/train_model.py
"""

import sys
import pickle
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append('..')

from src.inverse.inference import InferenceEngine
from src.core.contracts import GrowthSequence, GrowthModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    print("=" * 60)
    print("Urban Growth Model Training")
    print("=" * 60)
    
    # Load sequence
    sequence_path = Path('../data/processed/sjc_sequence_fine.pkl')
    print(f"\n📂 Loading growth sequence from {sequence_path}...")
    
    with open(sequence_path, 'rb') as f:
        sequence = pickle.load(f)
    
    print(f"   City: {sequence.city_name}")
    print(f"   States: {len(sequence.states)}")
    print(f"   Initial blocks: {len(sequence.initial_state.blocks)}")
    print(f"   Final blocks: {len(sequence.final_state.blocks)}")
    
    # ======================== FIX ISSUES ========================
    print(f"\n🔧 Preparing sequence for training...")
    
    # 1. Verify sequence is in forward (growth) order
    initial_blocks = len(sequence.initial_state.blocks)
    final_blocks = len(sequence.final_state.blocks)

    if final_blocks < initial_blocks:
        print(f"   ❌ ERROR: Sequence is in REVERSE order (rewind: {initial_blocks} → {final_blocks})")
        print(f"      rewind.py should output forward-ordered sequences!")
        sys.exit(1)
    else:
        print(f"   ✅ Sequence in forward (growth) order: {initial_blocks} → {final_blocks} blocks")
    
    # 2. Validate CRS - rewind.py should output metric CRS
    if sequence.states[0].blocks.crs.is_geographic:
        print(f"   ❌ ERROR: Sequence is in geographic CRS (lat/lon degrees)")
        print(f"      rewind.py should output metric CRS sequences!")
        print(f"      Current CRS: {sequence.states[0].blocks.crs}")
        sys.exit(1)
    else:
        print(f"   ✅ Sequence in metric CRS: {sequence.states[0].blocks.crs}")
    
    # 3. Verify growth is detectable
    blocks_delta = len(sequence.final_state.blocks) - len(sequence.initial_state.blocks)
    avg_growth = blocks_delta / (len(sequence.states) - 1) if len(sequence.states) > 1 else 0
    
    print(f"   📊 Total growth: {blocks_delta:+d} blocks across {len(sequence.states)} states")
    print(f"   📊 Avg growth per state: {avg_growth:.2f} blocks")
    
    if blocks_delta <= 0:
        print(f"   ❌ ERROR: No growth detected in sequence!")
        print(f"      Check if sequence was properly reversed.")
        sys.exit(1)
    
    # 4. Sample data verification
    sample_state = sequence.states[0]
    sample_block = sample_state.blocks.iloc[0]
    sample_area = sample_block.geometry.area
    
    print(f"   📊 Sample block area: {sample_area:.2f} m²")
    if sample_area < 1:
        print(f"   ⚠️  WARNING: Block areas still very small! CRS conversion may have failed.")
    else:
        print(f"   ✅ Block areas look reasonable (in m²)")
    
    print(f"✅ Sequence ready for training\n")
    # ============================================================
    
    # Train models
    print("🚀 Starting training...")
    engine = InferenceEngine(model_type='random_forest', random_state=42)
    
    try:
        growth_model = engine.train(sequence, test_size=0.2, verbose=True)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return
    
    # Save trained model
    output_dir = Path('../models')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    model_path = output_dir / 'sjc_growth_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(growth_model, f)
    
    print(f"\n💾 Model saved to: {model_path}")
    
    # Display feature importance
    print("\n🔍 Top 10 Most Important Features:")
    avg_importance = growth_model.training_metadata['feature_importance']['average']
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(sorted_features[:10], 1):
        print(f"   {i:2d}. {feature:20s}: {importance:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    metrics = growth_model.training_metadata['metrics']
    print(f"Location R² (X):       {metrics['location_r2_x']:.4f}")
    print(f"Location R² (Y):       {metrics['location_r2_y']:.4f}")
    print(f"Mean Spatial Error:    {metrics['mean_spatial_error_m']:.2f} m")
    print(f"Size R²:               {metrics['size_r2']:.4f}")
    print(f"Circularity R²:        {metrics['circularity_r2']:.4f}")
    print(f"Aspect Ratio R²:       {metrics['aspect_ratio_r2']:.4f}")
    print("=" * 60)
    
    # Optional: Create feature importance plot
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        features = [f[0] for f in sorted_features[:10]]
        importances = [f[1] for f in sorted_features[:10]]
        plt.barh(features, importances)
        plt.xlabel('Average Feature Importance')
        plt.title(f'Top 10 Features - {sequence.city_name} Urban Growth Model')
        plt.tight_layout()
        
        plot_path = output_dir / 'feature_importance.png'
        plt.savefig(plot_path, dpi=150)
        print(f"\n📊 Feature importance plot saved to: {plot_path}")
    except ImportError:
        logger.warning("matplotlib not available, skipping plot generation")
    
    print("\n✅ Training complete!")


if __name__ == '__main__':
    main()
