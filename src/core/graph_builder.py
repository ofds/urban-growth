"""
Converts OSM street networks to city block polygons.

The key insight: shapely.polygonize() requires fully noded networks
where every street intersection has an explicit vertex [web:43][web:44].
"""

import logging
from typing import List

import geopandas as gpd
import networkx as nx
import shapely
from shapely.geometry import LineString, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union, polygonize

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Converts OSM data to GrowthState components."""
    
    def extract_blocks_from_network(
        self,
        edges_gdf: gpd.GeoDataFrame,
        buffer_distance: float = 1.0
    ) -> gpd.GeoDataFrame:
        """
        Extract city blocks as polygons formed by street network.
        
        Algorithm [web:43][web:44][web:47]:
        1. Project to metric CRS if needed
        2. Collect all street LineStrings
        3. Node the network (ensure vertices at all intersections)
        4. Polygonize to create blocks
        5. Filter artifacts (too small, too large, sliver shapes)
        6. Convert back to original CRS
        
        Args:
            edges_gdf: GeoDataFrame with street geometries (LineStrings)
            buffer_distance: Tolerance for snapping intersections (in CRS units)
            
        Returns:
            GeoDataFrame of city block polygons
        """
        if len(edges_gdf) == 0:
            logger.warning("No streets provided - cannot extract blocks")
            return gpd.GeoDataFrame(geometry=[], crs=edges_gdf.crs)
        
        # Store original CRS
        original_crs = edges_gdf.crs
        
        # ========== CRITICAL FIX: Project to metric CRS ========== #
        if edges_gdf.crs.is_geographic:
            logger.info(f"Converting from {edges_gdf.crs} to local metric CRS...")
            edges_gdf = edges_gdf.to_crs(edges_gdf.estimate_utm_crs())
            logger.info(f"  → Using {edges_gdf.crs}")
        # ========================================================== #
        
        logger.info(f"Extracting blocks from {len(edges_gdf)} street segments...")
        
        # Step 1: Get all street geometries [web:44]
        lines = edges_gdf.geometry.tolist()
        
        # Step 2: Combine into single geometry collection [web:43]
        try:
            collection = shapely.GeometryCollection(lines)
        except:
            from shapely.geometry import GeometryCollection
            collection = GeometryCollection(lines)
        
        # Step 3: Node the network (critical step!) [web:43][web:47]
        logger.info("Noding network (adding vertices at intersections)...")
        try:
            noded = shapely.node(collection)
        except AttributeError:
            logger.warning("shapely.node() not available, using unary_union as fallback")
            noded = unary_union(lines)
        
        # Step 4: Polygonize to create blocks [web:41][web:42]
        logger.info("Polygonizing street network into blocks...")
        try:
            # Extract individual geometries from noded result
            if hasattr(noded, 'geoms'):
                geoms = noded.geoms
            else:
                geoms = [noded]
            
            # Polygonize
            polygons = list(polygonize(geoms))
            logger.info(f"Created {len(polygons)} raw polygons")
            
        except Exception as e:
            logger.error(f"Polygonization failed: {e}")
            return gpd.GeoDataFrame(geometry=[], crs=original_crs)
        
        if len(polygons) == 0:
            logger.warning("Polygonization produced 0 blocks - check street network connectivity")
            return gpd.GeoDataFrame(geometry=[], crs=original_crs)
        
        # Step 5: Filter artifacts [web:44][web:46]
        filtered_blocks = self._filter_block_artifacts(polygons, edges_gdf)
        logger.info(f"After filtering: {len(filtered_blocks)} valid blocks")
        
        # Step 6: Create GeoDataFrame with attributes
        blocks_gdf = gpd.GeoDataFrame(
            {
                'block_id': range(len(filtered_blocks)),
                'area': [p.area for p in filtered_blocks],
                'perimeter': [p.length for p in filtered_blocks],
            },
            geometry=filtered_blocks,
            crs=edges_gdf.crs  # This is now UTM
        )
        
        # ========== Convert back to original CRS ========== #
        if original_crs != blocks_gdf.crs:
            logger.info(f"Converting blocks back to {original_crs}")
            blocks_gdf = blocks_gdf.to_crs(original_crs)
        # ================================================== #
        
        return blocks_gdf


    def _filter_block_artifacts(
        self,
        polygons: List[Polygon],
        streets_gdf: gpd.GeoDataFrame
    ) -> List[Polygon]:
        """
        Remove artifact polygons [web:44][web:46].
        
        Assumes streets_gdf is in METRIC CRS (meters).
        """
        if len(polygons) == 0:
            return []
        
        # Calculate area statistics
        areas = [p.area for p in polygons]
        
        # ========== ADAPTIVE THRESHOLDS ========== #
        # Min: typical parking space is ~12 m², so 50 m² is reasonable minimum
        min_area = 50  # m²
        
        # Max: 99th percentile (removes outer boundary artifacts)
        sorted_areas = sorted(areas)
        if len(sorted_areas) > 10:
            max_area = sorted_areas[int(len(sorted_areas) * 0.99)]
        else:
            max_area = max(areas) * 2  # Allow everything if few polygons
        
        # Also filter by aspect ratio to remove slivers [web:43]
        filtered = []
        for p in polygons:
            if not (min_area <= p.area <= max_area and p.is_valid):
                continue
            
            # Remove sliver polygons (long, thin artifacts)
            # Circularity: 1.0 = perfect circle, 0 = line
            circularity = (4 * 3.14159 * p.area) / (p.length ** 2)
            if circularity < 0.05:  # Very elongated
                continue
            
            filtered.append(p)
        # ======================================== #
        
        logger.info(f"Filtered out {len(polygons) - len(filtered)} artifacts")
        logger.info(f"  - Area range: {min_area:.1f} to {max_area:.1f} m²")
        logger.info(f"  - Mean block area: {sum([p.area for p in filtered])/len(filtered):.1f} m²" if filtered else "")
        
        return filtered
