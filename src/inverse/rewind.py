"""
Rewind algorithm: Deconstructs final city state into plausible growth sequence.
ULTRA-OPTIMIZED VERSION - Keeps sequence in METRIC CRS for ML training
"""

import logging
from typing import List, Tuple, Dict, Any, Set, Optional, Iterator
from dataclasses import dataclass
import gc
import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import Point
import pandas as pd
from tqdm import tqdm

from src.core.contracts import GrowthState, GrowthSequence

logger = logging.getLogger(__name__)


@dataclass
class BlockRemovalCandidate:
    """Represents a block candidate for removal during rewind."""
    block_id: int
    score: float
    features: Dict[str, float]


class NeighborCache:
    """
    Maintains incremental neighbor relationships using spatial indexing.
    Optimization: O(n log n) queries instead of O(n²).
    """
    
    def __init__(self, blocks: gpd.GeoDataFrame):
        """Initialize with spatial index."""
        self.blocks = blocks
        self._build_index()
        self._neighbor_cache = {}
        self._removed_blocks = set()
    
    def _build_index(self):
        """Build STRtree spatial index."""
        if len(self.blocks) > 0:
            self.spatial_index = self.blocks.sindex
        else:
            self.spatial_index = None
    
    def get_neighbors(self, block_idx: int) -> Set[int]:
        """Get neighbors using spatial index O(log n + k)."""
        if block_idx in self._removed_blocks or block_idx not in self.blocks.index:
            return set()
        
        # Return cached if available
        if block_idx in self._neighbor_cache:
            return self._neighbor_cache[block_idx] - self._removed_blocks
        
        block = self.blocks.loc[block_idx]
        
        # Query spatial index
        if self.spatial_index is not None:
            possible_matches_idx = list(self.spatial_index.intersection(block.geometry.bounds))
            
            # Filter to actual touching neighbors
            neighbors = set()
            for idx in possible_matches_idx:
                if idx != block_idx and idx in self.blocks.index and idx not in self._removed_blocks:
                    if self.blocks.loc[idx].geometry.touches(block.geometry):
                        neighbors.add(idx)
        else:
            neighbors = set()
        
        self._neighbor_cache[block_idx] = neighbors.copy()
        return neighbors
    
    def remove_block(self, block_idx: int):
        """Mark block as removed (incremental update)."""
        self._removed_blocks.add(block_idx)
        
        if block_idx in self._neighbor_cache:
            neighbors = self._neighbor_cache[block_idx]
            # Invalidate neighbor caches
            for neighbor_idx in neighbors:
                if neighbor_idx in self._neighbor_cache:
                    del self._neighbor_cache[neighbor_idx]
            del self._neighbor_cache[block_idx]
    
    def get_all_neighbor_counts(self) -> Dict[int, int]:
        """Get neighbor counts for all active blocks."""
        counts = {}
        for idx in self.blocks.index:
            if idx not in self._removed_blocks:
                counts[idx] = len(self.get_neighbors(idx))
        return counts


class RewindEngine:
    """
    Reconstructs urban growth sequence by working backwards from final state.
    ULTRA-OPTIMIZED with generator-based lazy evaluation.
    """
    
    def __init__(
        self,
        target_ratio: float = 0.25,
        removal_strategy: str = 'composite',
        batch_size: int = 1,
        betweenness_sample_size: int = 100,
        snapshot_interval: int = 10
    ):
        """
        Args:
            target_ratio: Stop when this fraction of blocks remain
            removal_strategy: 'distance', 'neighbors', 'betweenness', or 'composite'
            batch_size: Remove multiple blocks per iteration
            betweenness_sample_size: Sample size for approximate betweenness
            snapshot_interval: Save full state every N steps (default: 10)
        """
        self.target_ratio = target_ratio
        self.removal_strategy = removal_strategy
        self.batch_size = batch_size
        self.betweenness_sample_size = betweenness_sample_size
        self.snapshot_interval = snapshot_interval
        self.neighbor_cache = None
        self.removed_blocks = set()
    
    def rewind(self, final_state: GrowthState) -> GrowthSequence:
        """
        Main entry point: Rewind final city state to initial seed.
        
        Returns:
            GrowthSequence with states in FORWARD order [initial → final]
            States are kept in METRIC CRS for ML training!
        """
        if len(final_state.blocks) == 0:
            raise ValueError("Cannot rewind empty city state")
        
        logger.info(f"Starting ULTRA-OPTIMIZED rewind from {len(final_state.blocks)} blocks")
        logger.info(f"Target: {int(len(final_state.blocks) * self.target_ratio)} blocks")
        logger.info(f"Batch size: {self.batch_size}, Snapshot interval: {self.snapshot_interval}")
        logger.info(f"Using GENERATOR-BASED lazy evaluation (minimal memory)")
        
        # ========== CONVERT TO METRIC CRS ========== #
        original_crs = final_state.streets.crs
        needs_crs_conversion = original_crs.is_geographic if original_crs else False
        
        if needs_crs_conversion:
            logger.info(f"Converting to metric CRS...")
            working_state = final_state.to_metric_crs()
            logger.info(f" → Using {working_state.streets.crs}")
        else:
            working_state = final_state.copy()
        
        # =========================================== #
        # Initialize neighbor cache
        self.neighbor_cache = NeighborCache(working_state.blocks)
        logger.info("✓ Spatial index built")
        
        # Pre-calculate city center (in METRIC CRS)
        self.city_center = working_state.blocks.geometry.union_all().centroid
        
        # Pre-calculate distances
        self._update_distance_cache(working_state)
        
        # Pre-calculate areas and perimeters
        self.areas = working_state.blocks.geometry.area
        self.perimeters = working_state.blocks.geometry.length
        
        logger.info("✓ Geometry properties cached")
        
        # Store only indices + pre-calculated frontiers
        snapshot_data = []
        initial_count = len(final_state.blocks)
        target_blocks = int(initial_count * self.target_ratio)
        step = 0
        total_steps = initial_count - target_blocks
        
        pbar = tqdm(total=total_steps, desc="Rewinding city", unit="blocks")
        
        # Save initial snapshot
        initial_indices = set(working_state.blocks.index)
        initial_frontiers = self._recalculate_frontiers_from_indices(working_state, initial_indices)
        snapshot_data.append((0, initial_indices, initial_frontiers))
        
        while self._get_active_block_count(working_state) > target_blocks:
            # Select block(s) to remove
            if self.batch_size > 1:
                candidates = self._select_batch_removal_candidates(
                    working_state,
                    min(self.batch_size, self._get_active_block_count(working_state) - target_blocks)
                )
            else:
                candidates = [self._select_removal_candidate(working_state)]
            
            # Remove blocks
            for candidate in candidates:
                # Update caches
                self.removed_blocks.add(candidate.block_id)
                self.neighbor_cache.remove_block(candidate.block_id)
                
                step += 1
                pbar.update(1)
                
                if step % 100 == 0:
                    active_count = self._get_active_block_count(working_state)
                    pbar.set_postfix({
                        'remaining': active_count,
                        'last_score': f"{candidate.score:.3f}"
                    })
            
            # Save snapshot with pre-calculated frontiers (in METRIC CRS)
            if step % self.snapshot_interval == 0:
                active_indices = set(idx for idx in working_state.blocks.index
                                   if idx not in self.removed_blocks)
                frontiers = self._recalculate_frontiers_from_indices(working_state, active_indices)
                snapshot_data.append((step, active_indices, frontiers))
        
        # Save final snapshot
        final_indices = set(idx for idx in working_state.blocks.index
                          if idx not in self.removed_blocks)
        final_frontiers = self._recalculate_frontiers_from_indices(working_state, final_indices)
        snapshot_data.append((step, final_indices, final_frontiers))
        
        pbar.close()
        logger.info(f"✓ Rewind complete: {step} steps, {len(snapshot_data)} snapshots")
        
        # ========== GENERATE SEQUENCE (KEEP IN METRIC CRS!) ========== #
        logger.info("Building sequence in METRIC CRS (for ML training)...")
        
        # Reverse snapshots for forward chronology
        snapshot_data.sort(key=lambda x: x[0])
        
        # *** KEY FIX: DON'T convert back to geographic! ***
        # Keep everything in METRIC CRS for ML training
        forward_sequence = list(self._generate_states_lazily(
            snapshot_data,
            working_state.blocks,  # Already metric
            working_state.streets,  # Already metric
            working_state.graph,
            convert_crs=False  # DON'T convert!
        ))
        
        logger.info(f"✓ ULTRA-OPTIMIZED rewind complete: {len(forward_sequence)} states")
        logger.info(f"✓ Sequence kept in METRIC CRS: {working_state.streets.crs}")
        
        return GrowthSequence(
            states=forward_sequence,
            city_name=final_state.metadata.get('city_name', 'unknown'),
            metadata={
                'rewind_strategy': self.removal_strategy,
                'target_ratio': self.target_ratio,
                'batch_size': self.batch_size,
                'snapshot_interval': self.snapshot_interval,
                'initial_blocks': initial_count,
                'final_blocks': len(final_indices),
                'original_crs': str(original_crs) if original_crs else 'metric',
                'output_crs': str(working_state.streets.crs),  # Document output CRS
                'ultra_optimized': True,
                'lazy_evaluation': True
            }
        )
    
    def _generate_states_lazily(
        self,
        snapshot_data: List[Tuple[int, Set[int], List[Point]]],
        base_blocks: gpd.GeoDataFrame,
        streets: gpd.GeoDataFrame,
        graph: nx.Graph,
        convert_crs: bool = False
    ) -> Iterator[GrowthState]:
        """
        Generator that yields states one at a time.
        Memory usage: Only 1-2 states in memory at once.
        
        Args:
            convert_crs: If True, convert to geographic (NOT recommended for ML)
        """
        pbar = tqdm(snapshot_data, desc="Generating states", unit="state")
        
        for i, (step, active_indices, frontiers) in enumerate(pbar):
            # Create state with active blocks only
            active_blocks = base_blocks.loc[list(active_indices)].copy()
            
            state = GrowthState(
                blocks=active_blocks,
                streets=streets,  # Shared reference
                graph=graph,  # Shared reference
                frontiers=frontiers,
                step=i * self.snapshot_interval,
                metadata={'removed_count': len(base_blocks) - len(active_indices)}
            )
            
            yield state
            
            # Force garbage collection every 10 states
            if (i + 1) % 10 == 0:
                gc.collect()
                pbar.set_postfix({'status': 'gc cleaned', 'state': i+1})
        
        pbar.close()
    
    def _recalculate_frontiers_from_indices(
        self,
        state: GrowthState,
        active_indices: Set[int]
    ) -> List[Point]:
        """
        Calculate frontiers from indices (in METRIC CRS to avoid warnings).
        """
        if len(active_indices) == 0:
            return []
        
        # Use pre-calculated distances
        active_distances = self.distance_cache[list(active_indices)]
        percentile_value = active_distances.quantile(0.8)
        peripheral_indices = active_distances[active_distances >= percentile_value].index
        
        # Get centroids in METRIC CRS (no warnings)
        peripheral_blocks = state.blocks.loc[peripheral_indices]
        frontiers = peripheral_blocks.geometry.centroid.tolist()
        
        return frontiers[:100]
    
    def _get_active_block_count(self, state: GrowthState) -> int:
        """Count blocks that haven't been logically removed."""
        return len([idx for idx in state.blocks.index if idx not in self.removed_blocks])
    
    def _get_active_blocks(self, state: GrowthState) -> gpd.GeoDataFrame:
        """Get only active (non-removed) blocks."""
        active_indices = [idx for idx in state.blocks.index if idx not in self.removed_blocks]
        return state.blocks.loc[active_indices]
    
    def _update_distance_cache(self, state: GrowthState):
        """Pre-calculate all distances (vectorized)."""
        if len(state.blocks) > 0:
            centroids = state.blocks.geometry.centroid
            self.distance_cache = centroids.distance(self.city_center)
            self.max_distance = self.distance_cache.max()
        else:
            self.distance_cache = pd.Series(dtype=float)
            self.max_distance = 1.0
    
    def _select_removal_candidate(self, state: GrowthState) -> BlockRemovalCandidate:
        """Choose which block to remove."""
        if self.removal_strategy == 'distance':
            return self._score_by_distance(state)
        elif self.removal_strategy == 'neighbors':
            return self._score_by_neighbors(state)
        elif self.removal_strategy == 'betweenness':
            return self._score_by_betweenness(state)
        else:
            return self._score_composite(state)
    
    def _select_batch_removal_candidates(
        self,
        state: GrowthState,
        batch_size: int
    ) -> List[BlockRemovalCandidate]:
        """Select multiple peripheral blocks."""
        all_scores = self._score_all_blocks(state)
        sorted_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)
        
        candidates = []
        for idx, score, features in sorted_scores[:batch_size]:
            candidates.append(BlockRemovalCandidate(
                block_id=idx,
                score=score,
                features=features
            ))
        
        return candidates
    
    def _score_all_blocks(self, state: GrowthState) -> List[Tuple[int, float, Dict]]:
        """Score all active blocks (vectorized)."""
        scores = []
        neighbor_counts = self.neighbor_cache.get_all_neighbor_counts()
        norm_distances = self.distance_cache / self.max_distance if self.max_distance > 0 else self.distance_cache
        
        for idx in state.blocks.index:
            if idx in self.removed_blocks:
                continue
            
            dist_score = norm_distances.loc[idx]
            neighbor_count = neighbor_counts.get(idx, 0)
            neighbor_score = 1.0 / (neighbor_count + 1)
            
            area = self.areas.loc[idx]
            perimeter = self.perimeters.loc[idx]
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            compact_score = 1.0 - circularity
            
            composite = (
                0.50 * dist_score +
                0.30 * neighbor_score +
                0.20 * compact_score
            )
            
            features = {
                'distance_to_center': self.distance_cache.loc[idx],
                'neighbor_count': neighbor_count,
                'area': area,
                'circularity': circularity,
                'composite_score': composite
            }
            
            scores.append((idx, composite, features))
        
        return scores
    
    def _score_by_distance(self, state: GrowthState) -> BlockRemovalCandidate:
        """Periphery: Distance from center."""
        scores = []
        for idx in state.blocks.index:
            if idx in self.removed_blocks:
                continue
            
            distance = self.distance_cache.loc[idx]
            features = {
                'distance_to_center': distance,
                'area': self.areas.loc[idx],
                'perimeter': self.perimeters.loc[idx]
            }
            scores.append((idx, distance, features))
        
        block_id, score, features = max(scores, key=lambda x: x[1])
        return BlockRemovalCandidate(block_id=block_id, score=score, features=features)
    
    def _score_by_neighbors(self, state: GrowthState) -> BlockRemovalCandidate:
        """Periphery: Few neighbors."""
        scores = []
        for idx in state.blocks.index:
            if idx in self.removed_blocks:
                continue
            
            neighbors = self.neighbor_cache.get_neighbors(idx)
            neighbor_count = len(neighbors)
            score = 1.0 / (neighbor_count + 1)
            
            features = {
                'neighbor_count': neighbor_count,
                'distance_to_center': self.distance_cache.loc[idx],
                'area': self.areas.loc[idx]
            }
            scores.append((idx, score, features))
        
        block_id, score, features = max(scores, key=lambda x: x[1])
        return BlockRemovalCandidate(block_id=block_id, score=score, features=features)
    
    def _score_by_betweenness(self, state: GrowthState) -> BlockRemovalCandidate:
        """Periphery: Low betweenness."""
        if state.graph.number_of_edges() > 0:
            try:
                k = min(self.betweenness_sample_size, state.graph.number_of_nodes())
                betweenness = nx.betweenness_centrality(state.graph, k=k, normalized=True)
            except Exception as e:
                logger.warning(f"Betweenness failed: {e}, using distance")
                return self._score_by_distance(state)
        else:
            return self._score_by_distance(state)
        
        scores = []
        for idx, block in state.blocks.iterrows():
            if idx in self.removed_blocks:
                continue
            
            block_nodes = []
            for node in state.graph.nodes():
                node_geom = Point(state.graph.nodes[node]['x'], state.graph.nodes[node]['y'])
                if block.geometry.buffer(0.0001).contains(node_geom):
                    block_nodes.append(node)
            
            if block_nodes:
                avg_betweenness = np.mean([betweenness.get(n, 0) for n in block_nodes])
                score = 1.0 / (avg_betweenness + 0.001)
            else:
                score = 1.0
            
            features = {
                'betweenness': 1.0 / score if score > 0 else 0,
                'distance_to_center': self.distance_cache.loc[idx],
                'area': self.areas.loc[idx]
            }
            scores.append((idx, score, features))
        
        block_id, score, features = max(scores, key=lambda x: x[1])
        return BlockRemovalCandidate(block_id=block_id, score=score, features=features)
    
    def _score_composite(self, state: GrowthState) -> BlockRemovalCandidate:
        """Composite scoring."""
        all_scores = self._score_all_blocks(state)
        block_id, score, features = max(all_scores, key=lambda x: x[1])
        return BlockRemovalCandidate(block_id=block_id, score=score, features=features)
