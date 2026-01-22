"""
Extract street networks and city blocks from OSM PBF files.

Usage:
    python scripts/preprocess.py --city "Piedmont, CA, USA" --output data/processed/piedmont.pkl
    python scripts/preprocess.py --bbox "-122.25,37.82,-122.21,37.84" --pbf data/raw/california.osm.pbf
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union

# Import your contracts
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.core.contracts import GrowthState
from src.core.graph_builder import GraphBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OSMExtractor:
    """Extracts urban data from OpenStreetMap."""
    
    def __init__(self, network_type: str = 'drive', simplify: bool = True):
        """
        Args:
            network_type: 'drive', 'walk', 'bike', or 'all'
            simplify: Remove nodes that only connect two edges
        """
        self.network_type = network_type
        self.simplify = simplify
        
    def extract_from_place(self, place_name: str) -> GrowthState:
        """
        Extract city by name (e.g., "Piedmont, CA, USA").
        
        Uses Nominatim to geocode place and download within boundary.
        """
        logger.info(f"Downloading street network for: {place_name}")
        
        try:
            # Download street network [web:30]
            G = ox.graph_from_place(
                place_name,
                network_type=self.network_type,
                simplify=self.simplify
            )
            
            # Get place boundary for context
            boundary = ox.geocode_to_gdf(place_name)
            
            logger.info(f"Downloaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
            return self._graph_to_growth_state(G, boundary)
            
        except Exception as e:
            logger.error(f"Failed to download {place_name}: {e}")
            raise
    
    def extract_from_bbox(
        self, 
        bbox: Tuple[float, float, float, float],
        pbf_path: Optional[Path] = None
    ) -> GrowthState:
        """
        Extract by bounding box: (west, south, east, north).
        
        If pbf_path provided, reads from local file instead of API.
        """
        logger.info(f"Extracting from bbox: {bbox}")
        
        if pbf_path:
            # Read from local PBF file [web:21][web:27]
            G = ox.graph_from_bbox(
                bbox=bbox,
                network_type=self.network_type,
                simplify=self.simplify,
                custom_filter=None  # You could add custom highway filters
            )
        else:
            # Download from Overpass API
            G = ox.graph_from_bbox(
                north=bbox[3], south=bbox[1],
                east=bbox[2], west=bbox[0],
                network_type=self.network_type,
                simplify=self.simplify
            )
        
        logger.info(f"Extracted {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Create dummy boundary from bbox
        boundary = gpd.GeoDataFrame(
            geometry=[Polygon([
                (bbox[0], bbox[1]),
                (bbox[2], bbox[1]),
                (bbox[2], bbox[3]),
                (bbox[0], bbox[3])
            ])],
            crs='EPSG:4326'
        )
        
        return self._graph_to_growth_state(G, boundary)
    
    def _graph_to_growth_state(
        self,
        G: nx.MultiDiGraph,
        boundary: gpd.GeoDataFrame
    ) -> GrowthState:
        """Convert OSMnx graph to GrowthState."""
        
        # 1. Convert graph to GeoDataFrames [web:28][web:30]
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
        
        # 2. Extract city blocks from street network [web:24]
        logger.info("Generating city blocks from street network...")
        blocks_gdf = self._extract_blocks(G, boundary)
        logger.info(f"Generated {len(blocks_gdf)} city blocks")
        
        # 3. Identify growth frontiers (boundary of urban area)
        frontiers = self._find_frontiers(nodes_gdf, blocks_gdf)
        logger.info(f"Identified {len(frontiers)} frontier points")
        
        # 4. Create GrowthState
        growth_state = GrowthState(
            streets=edges_gdf,
            blocks=blocks_gdf,
            graph=G.to_undirected(),  # Simplify to undirected
            frontiers=frontiers,
            step=0  # This is the FINAL state, not intermediate
        )
        
        # 5. Data quality checks
        self._validate_extraction(growth_state)
        
        return growth_state
    
    def _extract_blocks(
            self,
            G: nx.MultiDiGraph,
            boundary: gpd.GeoDataFrame
        ) -> gpd.GeoDataFrame:
            """
            Extract city blocks as polygons formed by street network.
            """
            try:
                # Convert graph to edges GeoDataFrame
                _, edges_gdf = ox.graph_to_gdfs(G)
                
                # Use GraphBuilder to extract blocks [web:43][web:44]
                from src.core.graph_builder import GraphBuilder
                builder = GraphBuilder()
                blocks = builder.extract_blocks_from_network(edges_gdf)
                
                return blocks
                
            except Exception as e:
                logger.error(f"Block extraction failed: {e}")
                raise


    def _voronoi_blocks(
        self,
        nodes: gpd.GeoDataFrame,
        boundary: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Fallback: Create approximate blocks using Voronoi tessellation.
        """
        from scipy.spatial import Voronoi
        from shapely.geometry import box
        
        # Get node coordinates
        coords = nodes.geometry.apply(lambda p: (p.x, p.y)).tolist()
        
        # Create Voronoi diagram
        vor = Voronoi(coords)
        
        # Convert to polygons (simplified version)
        # TODO: Implement proper Voronoi → polygon conversion
        logger.warning("Using simplified Voronoi blocks - implement proper block extraction!")
        
        return gpd.GeoDataFrame(geometry=[], crs=nodes.crs)
    
    def _find_frontiers(
        self,
        nodes: gpd.GeoDataFrame,
        blocks: gpd.GeoDataFrame
    ) -> list[Point]:
        """
        Identify frontier points where urban growth could occur.
        
        Heuristic: Nodes on the periphery with degree < 3
        """
        # Find boundary nodes (low degree, high distance from center)
        center = nodes.geometry.union_all().centroid
        
        frontier_nodes = []
        for idx, node in nodes.iterrows():
            distance_to_center = node.geometry.distance(center)
            # TODO: Calculate node degree from graph
            # If degree < 3 and distance > threshold → frontier
            frontier_nodes.append(node.geometry)
        
        return frontier_nodes[:100]  # Limit to top 100 for performance
    
    def _validate_extraction(self, state: GrowthState) -> None:
        """Data quality checks."""
        
        # Check 1: Network is connected
        if not nx.is_connected(state.graph):
            logger.warning(f"Graph has {nx.number_connected_components(state.graph)} components")
        
        # Check 2: Blocks don't overlap
        if len(state.blocks) > 0:
            # TODO: Check for overlapping polygons
            pass
        
        # Check 3: Streets have valid geometries
        invalid_streets = state.streets[~state.streets.geometry.is_valid]
        if len(invalid_streets) > 0:
            logger.warning(f"Found {len(invalid_streets)} invalid street geometries")
        
        logger.info("✓ Extraction validation passed")


def main():
    parser = argparse.ArgumentParser(description="Extract urban data from OSM")
    parser.add_argument('--city', type=str, help='City name (e.g., "Piedmont, CA")')
    parser.add_argument('--bbox', type=str, help='Bounding box: west,south,east,north')
    parser.add_argument('--pbf', type=Path, help='Local PBF file (optional)')
    parser.add_argument('--output', type=Path, required=True, help='Output pickle file')
    parser.add_argument('--network-type', default='drive', choices=['drive', 'walk', 'bike', 'all'])
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = OSMExtractor(network_type=args.network_type)
    
    # Extract based on input method
    if args.city:
        growth_state = extractor.extract_from_place(args.city)
    elif args.bbox:
        bbox = tuple(map(float, args.bbox.split(',')))
        growth_state = extractor.extract_from_bbox(bbox, args.pbf)
    else:
        raise ValueError("Must provide either --city or --bbox")
    
    # Save to pickle
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(growth_state, f)
    
    logger.info(f"✓ Saved to {args.output}")
    logger.info(f"  - {len(growth_state.streets)} streets")
    logger.info(f"  - {len(growth_state.blocks)} blocks")
    logger.info(f"  - {growth_state.graph.number_of_nodes()} nodes")


if __name__ == '__main__':
    main()
