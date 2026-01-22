"""
Diagnose training data issues.
"""
import pickle
import numpy as np
import sys
sys.path.append('..')

from src.core.contracts import GrowthSequence

# Load sequence
with open('data/processed/sjc_sequence.pkl', 'rb') as f:
    sequence = pickle.load(f)

print("=" * 60)
print("🔍 DIAGNOSTIC REPORT")
print("=" * 60)

# 1. Check sequence order
print(f"\n1. Sequence Order:")
print(f"   Initial blocks: {len(sequence.initial_state.blocks)}")
print(f"   Final blocks:   {len(sequence.final_state.blocks)}")
print(f"   Direction:      {'✅ Forward (growth)' if len(sequence.final_state.blocks) > len(sequence.initial_state.blocks) else '❌ Reverse (rewind)'}")

# 2. Check CRS
print(f"\n2. Coordinate System:")
state = sequence.states[0]
print(f"   CRS:           {state.blocks.crs}")
print(f"   Is Geographic: {state.blocks.crs.is_geographic}")
if state.blocks.crs.is_geographic:
    print(f"   ⚠️  WARNING: Using lat/lon degrees, not meters!")
    print(f"   → Locations will be tiny numbers like (-23.1234, -45.8765)")
else:
    print(f"   ✅ Using metric CRS (good!)")

# 3. Sample block coordinates and sizes
print(f"\n3. Sample Data (State 0):")
sample_block = state.blocks.iloc[0]
centroid = sample_block.geometry.centroid
area = sample_block.geometry.area
print(f"   Sample Block Centroid: ({centroid.x:.6f}, {centroid.y:.6f})")
print(f"   Sample Block Area:     {area:.6f}")
if state.blocks.crs.is_geographic:
    print(f"   ⚠️  Area in square degrees (useless for ML!)")
else:
    print(f"   ✅ Area in m² (good!)")

# 4. Check block changes between states
print(f"\n4. Block Changes Between States:")
changes = []
for i in range(min(10, len(sequence.states) - 1)):
    before = len(sequence.states[i].blocks)
    after = len(sequence.states[i+1].blocks)
    delta = after - before
    changes.append(delta)
    print(f"   State {i:3d} → {i+1:3d}: {before:4d} → {after:4d} blocks (Δ = {delta:+4d})")

if all(d == 0 for d in changes):
    print(f"   ❌ NO CHANGES DETECTED! Snapshot interval too large?")
elif all(d > 0 for d in changes):
    print(f"   ✅ Consistent growth detected")
else:
    print(f"   ⚠️  Mixed changes (some negative?)")

# 5. Label extraction test
print(f"\n5. Label Extraction Test:")
from src.inverse.inference import InferenceEngine
engine = InferenceEngine()

state_before = sequence.states[0]
state_after = sequence.states[1]
labels = engine._extract_labels(state_before, state_after)

if labels is None:
    print(f"   ❌ CRITICAL: No labels extracted!")
    print(f"      Blocks before: {len(state_before.blocks)}")
    print(f"      Blocks after:  {len(state_after.blocks)}")
    print(f"      → Are consecutive states identical?")
else:
    print(f"   ✅ Labels extracted successfully:")
    print(f"      Location:    ({labels['location'][0]:.6f}, {labels['location'][1]:.6f})")
    print(f"      Size:        {labels['size']:.6f}")
    print(f"      Circularity: {labels['shape'][0]:.4f}")
    print(f"      Aspect:      {labels['shape'][1]:.4f}")

# 6. Feature variance check
print(f"\n6. Feature Variance Check:")
features_list = []
for i in range(min(50, len(sequence.states))):
    features = engine._extract_features(sequence.states[i])
    features_list.append(features)

X = np.array(features_list)
feature_names = engine._get_feature_names()

print(f"   Feature statistics (first 50 states):")
for i, name in enumerate(feature_names[:5]):  # Show first 5
    mean = X[:, i].mean()
    std = X[:, i].std()
    print(f"   {name:20s}: mean={mean:.4f}, std={std:.4f}")
    if std < 1e-6:
        print(f"      ⚠️  NO VARIANCE - feature is constant!")

print("\n" + "=" * 60)
