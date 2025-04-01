#!/usr/bin/env python3
"""
Weighted region clustering for newspaper image analysis.
Clusters newspaper images based on the semantic similarity of their sub-regions,
weighted by the area each region occupies in the image.
"""

import os
import json
import datetime
import numpy as np
import logging
from collections import defaultdict
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import networkx as nx

from config import (
    IMAGE_FOLDER, OUTPUT_FOLDER, REGION_TYPES_TO_PROCESS,
    REGION_SIMILARITY_THRESHOLD, WEIGHT_BY_AREA
)
from logger_setup import logger
from db_operations import initialize_db
from doclayout_detector import DocLayoutDetector, process_image_regions
from embedder import MmE5MllamaEmbedder
from image_utils import get_image_paths
from progress_tracker import (
    load_region_comparison_progress, save_region_comparison_progress,
    mark_region_comparison_as_completed, is_region_comparison_completed
)

# Create output folder for weighted clustering results
WEIGHTED_CLUSTERING_FOLDER = os.path.join(OUTPUT_FOLDER, "weighted_clustering")
WEIGHTED_CLUSTERING_PROGRESS_FILE = os.path.join(OUTPUT_FOLDER, "weighted_clustering_progress.json")

def load_clustering_progress():
    """Load the weighted clustering progress from a JSON file."""
    if os.path.exists(WEIGHTED_CLUSTERING_PROGRESS_FILE):
        try:
            with open(WEIGHTED_CLUSTERING_PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading weighted clustering progress file: {str(e)}")
            return {"completed_comparisons": []}
    return {"completed_comparisons": []}

def save_clustering_progress(progress_data):
    """Save the weighted clustering progress to a JSON file."""
    os.makedirs(os.path.dirname(WEIGHTED_CLUSTERING_PROGRESS_FILE), exist_ok=True)
    
    try:
        with open(WEIGHTED_CLUSTERING_PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f)
    except Exception as e:
        logger.error(f"Error saving weighted clustering progress file: {str(e)}")

def mark_clustering_as_completed(image_pair):
    """Mark a specific image pair as having completed clustering."""
    progress = load_clustering_progress()
    
    if image_pair not in progress["completed_comparisons"]:
        progress["completed_comparisons"].append(image_pair)
        save_clustering_progress(progress)

def is_clustering_completed(image_pair):
    """Check if clustering has already been done for a specific image pair."""
    progress = load_clustering_progress()
    return image_pair in progress["completed_comparisons"]

def safe_query(collection, query_embedding, n_results, where_clause, max_retries=3):
    """
    Safely perform a query with retries and error handling.
    """
    for attempt in range(max_retries):
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["metadatas", "documents", "distances"],
                where=where_clause
            )
            return results
        except RuntimeError as e:
            if "Cannot return the results in a contigious 2D array" in str(e):
                logger.warning(f"Query failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    # Reduce the number of results requested
                    n_results = max(1, int(n_results * 0.8))
                    logger.info(f"Retrying with reduced n_results={n_results}")
                    continue
            raise
    return None

def compute_image_similarity_matrix(collection, image_paths, similarity_threshold=REGION_SIMILARITY_THRESHOLD, skip_same_prefix=True, prefix_length=20):
    """
    Compute a similarity matrix between all pairs of newspaper images based on
    the similarity of their sub-regions, weighted by region area.
    """
    # Initialize variables
    image_names = [os.path.basename(path) for path in image_paths]
    n_images = len(image_names)
    similarity_matrix = np.zeros((n_images, n_images))
    
    # Create lookup for image index by name
    image_name_to_idx = {name: idx for idx, name in enumerate(image_names)}
    
    # Get all image regions from the database
    all_entries = collection.get(
        include=["metadatas", "embeddings"],
        where={"is_region": {"$eq": True}}
    )
    
    if not all_entries or len(all_entries["metadatas"]) == 0:
        logger.warning("No regions found in the database. Make sure regions have been processed first.")
        return None, None
    
    # Group regions by parent image
    image_to_regions = defaultdict(list)
    region_embeddings = {}
    region_areas = {}
    region_types = {}
    
    for i, region_id in enumerate(all_entries["ids"]):
        metadata = all_entries["metadatas"][i]
        embedding = all_entries["embeddings"][i]
        
        if metadata and embedding is not None and len(embedding) > 0:
            parent_image = metadata.get("parent_image_name")
            area_percentage = metadata.get("area_percentage", 0)
            region_type = metadata.get("region_type")
            
            if parent_image and area_percentage > 0 and region_type in REGION_TYPES_TO_PROCESS:
                image_to_regions[parent_image].append(region_id)
                region_embeddings[region_id] = embedding
                region_areas[region_id] = area_percentage / 100.0
                region_types[region_id] = region_type
    
    # Compute cross-image region similarities
    logger.info("Computing cross-image region similarities...")
    
    total_pairs = (n_images * (n_images - 1)) // 2
    
    # Calculate the total number of regions for logging
    total_regions = sum(len(regions) for regions in image_to_regions.values())
    logger.info(f"Found {total_regions} regions across {n_images} images")
    
    # Set a VERY permissive threshold for initial clustering - try to find ANY connections
    effective_threshold = 0.1  # Much more permissive than default (0.7)
    logger.info(f"Using similarity threshold: {effective_threshold} (original: {similarity_threshold})")

    # Track statistics for debugging
    total_comparisons = 0
    passing_comparisons = 0
    region_matches_by_pair = defaultdict(int)
    distance_values = []  # For diagnostic histogram of distances
    
    with tqdm(total=total_pairs, desc="Computing image similarities") as pbar:
        # Process each image pair
        for i in range(n_images):
            img_i = image_names[i]
            regions_i = image_to_regions.get(img_i, [])
            
            if not regions_i:
                logger.debug(f"Image {img_i} has no regions, skipping")
                continue
                
            for j in range(i+1, n_images):  # Only compute upper triangle to avoid duplicates
                img_j = image_names[j]
                regions_j = image_to_regions.get(img_j, [])
                
                if not regions_j:
                    logger.debug(f"Image {img_j} has no regions, skipping")
                    continue
                
                # Skip pairs with same prefix if enabled
                if skip_same_prefix:
                    source_prefix = img_i[:min(prefix_length, len(img_i))]
                    target_prefix = img_j[:min(prefix_length, len(img_j))]
                    
                    if source_prefix == target_prefix:
                        logger.debug(f" - Skipping comparison between {img_i} and {img_j} - identical prefix '{source_prefix}'")
                        pbar.update(1)
                        continue
                
                # Check if this pair has already been processed
                pair_key = f"{img_i}_{img_j}"
                if is_clustering_completed(pair_key):
                    pbar.update(1)
                    continue
                
                # For each region in image i, find its similarity to all regions in image j
                weighted_similarities = []
                pair_distances = []  # Track distances for this pair
                
                # Process similar regions
                for region_i in regions_i[:10]:  # Limit to first 10 regions for performance
                    embedding_i = region_embeddings.get(region_i)
                    area_i = region_areas.get(region_i, 0)
                    
                    if embedding_i is None or len(embedding_i) == 0 or area_i == 0:
                        continue
                    
                    # Use safe_query instead of direct collection.query
                    results = safe_query(
                        collection,
                        embedding_i,
                        n_results=min(10, len(regions_j)),
                        where_clause={"parent_image_name": {"$eq": img_j}}
                    )
                    
                    if not results:
                        continue
                        
                    # Process results
                    for k, distance in enumerate(results["distances"][0]):
                        if "metadatas" in results and k < len(results["metadatas"][0]):
                            metadata_j = results["metadatas"][0][k]
                            area_j = metadata_j.get("area_percentage", 0) / 100.0
                            
                            if distance <= (1.0 - effective_threshold) and area_j > 0:
                                similarity_score = 1.0 - distance
                                weighted_similarity = similarity_score * area_i * area_j
                                weighted_similarities.append(weighted_similarity)
                                passing_comparisons += 1
                                region_matches_by_pair[pair_key] += 1
                
                # Compute overall weighted similarity between the two images
                if weighted_similarities:
                    similarity_score = np.sum(weighted_similarities)
                    similarity_matrix[i, j] = similarity_score
                    similarity_matrix[j, i] = similarity_score  # Make matrix symmetric
                    
                    logger.info(f"Similarity between {img_i} and {img_j}: {similarity_score:.6f} " +
                              f"(based on {len(weighted_similarities)} matching region pairs)")
                else:
                    logger.debug(f"No similar regions found between {img_i} and {img_j}")
                
                # Mark pair as completed
                mark_clustering_as_completed(pair_key)
                pbar.update(1)
    
    # Normalize similarity matrix (excluding diagonal)
    max_non_diagonal = np.max(similarity_matrix - np.diag(np.diag(similarity_matrix)))
    if max_non_diagonal > 0:
        non_diag_mask = ~np.eye(n_images, dtype=bool)
        similarity_matrix[non_diag_mask] = similarity_matrix[non_diag_mask] / max_non_diagonal
    
    # Set diagonal to 1.0 (self-similarity)
    np.fill_diagonal(similarity_matrix, 1.0)
    
    return similarity_matrix, image_names

def plot_similarity_heatmap(similarity_matrix, image_names, output_path=None):
    """
    Plot a heatmap of the image similarity matrix.
    
    Args:
        similarity_matrix: Similarity matrix between all pairs of images
        image_names: List of image names corresponding to matrix rows/columns
        output_path: Path to save the plot
    """
    plt.figure(figsize=(14, 12))
    
    # Create heatmap with modified colormap to highlight non-zero values
    cmap = plt.cm.viridis.copy()
    cmap.set_under('black')  # Values below vmin will be black
    
    # Calculate a small positive value for vmin (to separate zeros from small similarities)
    min_nonzero = similarity_matrix[similarity_matrix > 0].min() if np.any(similarity_matrix > 0) else 0.01
    vmin = min(0.01, min_nonzero / 2)
    
    plt.imshow(similarity_matrix, cmap=cmap, interpolation="nearest", vmin=vmin)
    
    # Add colorbar and labels
    plt.colorbar(label="Similarity Score")
    plt.title("Newspaper Image Similarity Heatmap")
    
    # Annotate axes with image names (shortened for readability)
    short_names = [name[:15] + "..." if len(name) > 18 else name for name in image_names]
    plt.xticks(range(len(image_names)), short_names, rotation=90, fontsize=8)
    plt.yticks(range(len(image_names)), short_names, fontsize=8)
    
    plt.tight_layout()
    
    # Save if output path is specified
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved similarity heatmap to {output_path}")
    else:
        plt.show()

def plot_dendrogram(similarity_matrix, image_names, output_path=None):
    """
    Plot a dendrogram showing hierarchical clustering of newspaper images.
    
    Args:
        similarity_matrix: Similarity matrix between all pairs of images
        image_names: List of image names corresponding to matrix rows/columns
        output_path: Path to save the plot
    """
    from scipy.cluster.hierarchy import linkage, dendrogram
    
    # Convert similarity to distance
    distance_matrix = 1.0 - similarity_matrix
    
    # Perform hierarchical clustering
    Z = linkage(distance_matrix, method='average')
    
    # Plot dendrogram
    plt.figure(figsize=(16, 10))
    
    # Create shortened names for readability
    short_names = [name[:15] + "..." if len(name) > 18 else name for name in image_names]
    
    dendrogram(
        Z,
        labels=short_names,
        orientation="top",
        leaf_rotation=90,
        leaf_font_size=8,
    )
    
    plt.title("Hierarchical Clustering of Newspaper Images")
    plt.xlabel("Newspapers")
    plt.ylabel("Distance")
    
    plt.tight_layout()
    
    # Save if output path is specified
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved dendrogram to {output_path}")
    else:
        plt.show()

def plot_similarity_network(similarity_matrix, image_names, threshold=0.3, output_path=None):
    """
    Plot a network graph of newspaper similarities.
    
    Args:
        similarity_matrix: Similarity matrix between all pairs of images
        image_names: List of image names corresponding to matrix rows/columns
        threshold: Minimum similarity to include an edge
        output_path: Path to save the plot
    """
    # For sparse data, use a lower threshold to ensure some connections are shown
    min_edge_count = 3  # Minimum number of edges we want to see
    
    # Dynamically adjust threshold if needed
    sim_values = similarity_matrix[~np.eye(similarity_matrix.shape[0], dtype=bool)]  # Get non-diagonal values
    sim_values = sim_values[sim_values > 0]  # Only consider non-zero similarities
    
    if len(sim_values) < min_edge_count:
        # Not enough connections even at the lowest threshold
        logger.warning(f"Very few connections in the similarity network (only {len(sim_values)} non-zero values)")
        # Use a very low threshold to show whatever connections exist
        threshold = 0.01 if len(sim_values) > 0 else 0
    elif np.sum(sim_values > threshold) < min_edge_count:
        # Lower the threshold to get at least min_edge_count connections
        if len(sim_values) >= min_edge_count:
            # Sort similarities and pick the threshold that gives us min_edge_count connections
            threshold = sorted(sim_values, reverse=True)[min(min_edge_count, len(sim_values))-1]
            logger.info(f"Adjusted network threshold to {threshold:.4f} to show at least {min_edge_count} connections")
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for i, name in enumerate(image_names):
        # Trim name for display
        short_name = name[:15] + "..." if len(name) > 18 else name
        G.add_node(i, name=short_name)
    
    # Add edges
    edges_added = 0
    for i in range(len(image_names)):
        for j in range(i+1, len(image_names)):
            similarity = similarity_matrix[i, j]
            if similarity > threshold:
                G.add_edge(i, j, weight=similarity)
                edges_added += 1
    
    logger.info(f"Created network with {len(image_names)} nodes and {edges_added} edges (threshold: {threshold:.4f})")
    
    # Create the plot
    plt.figure(figsize=(14, 14))
    
    # Position nodes using force-directed layout
    # Adjust k based on number of nodes to avoid overcrowding
    k_value = 0.3 + (0.5 / max(1, len(image_names)/10))
    pos = nx.spring_layout(G, k=k_value, iterations=100, seed=42)
    
    # Calculate node degrees AFTER adding all edges
    node_degrees = dict(G.degree())
    node_sizes = [300 + 100*node_degrees.get(n, 0) for n in G.nodes()]
    node_colors = [node_degrees.get(n, 0) for n in G.nodes()]
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes, 
        node_color=node_colors,
        cmap=plt.cm.viridis,
        alpha=0.8
    )
    
    # Draw edges with width proportional to similarity
    if edges_added > 0:
        edges = G.edges()
        weights = [G[u][v]["weight"] * 5 for u, v in edges]  # Scale for visibility
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, alpha=0.5)
    
    # Draw node labels with size based on degree
    label_sizes = {n: 8 + min(4, node_degrees.get(n, 0)) for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, 
        labels=nx.get_node_attributes(G, "name"), 
        font_size=10,
        font_weight='bold'
    )
    
    # Add a colorbar only if we have nodes with connections
    if edges_added > 0 and len(node_colors) > 0 and max(node_colors) > 0:
        # Create a custom axis for the colorbar
        plt.subplots_adjust(right=0.85)  # Make room for colorbar on right
        cbar_ax = plt.axes([0.88, 0.1, 0.03, 0.8])  # [left, bottom, width, height]
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(min(node_colors), max(node_colors)))
        sm.set_array([])
        plt.colorbar(sm, cax=cbar_ax, label="Node Degree (Number of Connections)")
    else:
        logger.info("No nodes with connections to display in colorbar")
    
    plt.title("Newspaper Similarity Network")
    plt.axis("off")
    
    # Save if output path is specified
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved similarity network to {output_path}")
    else:
        plt.show()

def cluster_images(similarity_matrix, image_names, n_clusters=None):
    """
    Cluster newspaper images based on their similarity matrix.
    """
    # Ensure diagonal is 1.0 (self-similarity)
    np.fill_diagonal(similarity_matrix, 1.0)
    
    # Convert similarity to distance
    distance_matrix = 1.0 - similarity_matrix
    
    # Validate distance matrix
    if not isinstance(distance_matrix, np.ndarray) or distance_matrix.size == 0:
        logger.error("Distance matrix is not a valid numpy array")
        return None
    
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        logger.error("Distance matrix is not square")
        return None
    
    if np.any(np.isnan(distance_matrix)):
        logger.error("Distance matrix contains NaN values")
        return None
    
    # Apply hierarchical clustering
    logger.info("Applying hierarchical clustering...")
    
    try:
        # If n_clusters is None, determine it automatically
        if n_clusters is None:
            # Count number of non-zero similarities
            nonzero_pairs = np.sum(similarity_matrix > 0.01) - similarity_matrix.shape[0]  # Exclude diagonal
            logger.info(f"Number of image pairs with non-zero similarity: {nonzero_pairs // 2}")  # Divide by 2 because matrix is symmetric
            
            # If there are very few connections, don't try to create too many clusters
            if nonzero_pairs < 10:
                max_clusters = min(3, len(image_names))
                logger.info(f"Few connections detected, limiting to max {max_clusters} clusters")
            else:
                max_clusters = min(10, len(image_names))
            
            best_score = -1
            best_n_clusters = 2
            
            for k in range(2, max_clusters + 1):
                # Check scikit-learn version and use appropriate parameters
                try:
                    # Try modern scikit-learn API first
                    clustering = AgglomerativeClustering(
                        n_clusters=k,
                        affinity='precomputed',
                        linkage='average'
                    ).fit(distance_matrix)
                except TypeError:
                    # Fall back to older scikit-learn API
                    clustering = AgglomerativeClustering(
                        n_clusters=k,
                        linkage='average'
                    ).fit(distance_matrix)
                
                unique_labels = np.unique(clustering.labels_)
                if len(unique_labels) > 1:
                    from sklearn.metrics import silhouette_score
                    try:
                        score = silhouette_score(distance_matrix, clustering.labels_, metric='precomputed')
                        
                        if score > best_score:
                            best_score = score
                            best_n_clusters = k
                            logger.info(f"Testing {k} clusters: silhouette score = {score:.4f} (new best)")
                        else:
                            logger.info(f"Testing {k} clusters: silhouette score = {score:.4f}")
                    except Exception as e:
                        logger.warning(f"Error calculating silhouette score for k={k}: {e}")
                        continue
            
            n_clusters = best_n_clusters
            logger.info(f"Automatically determined optimal number of clusters: {n_clusters} (score: {best_score:.4f})")
        
        # Perform final clustering
        try:
            # Try modern scikit-learn API first
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                linkage='average'
            ).fit(distance_matrix)
        except TypeError:
            # Fall back to older scikit-learn API
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='average'
            ).fit(distance_matrix)
        
        # Group images by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(clustering.labels_):
            clusters[int(label)].append(image_names[i])
        
        # Calculate intra-cluster similarity (cohesion)
        cluster_cohesion = {}
        for label, images in clusters.items():
            indices = [image_names.index(img) for img in images]
            if len(indices) > 1:
                submatrix = similarity_matrix[np.ix_(indices, indices)]
                # Extract off-diagonal elements only
                mask = ~np.eye(submatrix.shape[0], dtype=bool)
                avg_similarity = np.mean(submatrix[mask]) if np.any(mask) else 0
                cluster_cohesion[label] = float(avg_similarity)
            else:
                cluster_cohesion[label] = 0.0
        
        return {
            "n_clusters": n_clusters,
            "clusters": {str(k): v for k, v in clusters.items()},
            "cluster_cohesion": cluster_cohesion,
            "labels": clustering.labels_.tolist()
        }
    
    except Exception as e:
        logger.error(f"Error during clustering: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def create_html_report(similarity_matrix, image_names, clustering_results, html_dir):
    """
    Create an HTML report with clustering results and similarity visualization.
    
    Args:
        similarity_matrix: Similarity matrix between all pairs of images
        image_names: List of image names corresponding to matrix rows/columns
        clustering_results: Dictionary with clustering information
        html_dir: Directory to save HTML files
    """
    os.makedirs(html_dir, exist_ok=True)
    
    # Create images directory for visualizations
    img_dir = os.path.join(html_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    # Generate visualizations
    heatmap_path = os.path.join(img_dir, "similarity_heatmap.png")
    dendrogram_path = os.path.join(img_dir, "clustering_dendrogram.png")
    network_path = os.path.join(img_dir, "similarity_network.png")
    
    plot_similarity_heatmap(similarity_matrix, image_names, heatmap_path)
    plot_dendrogram(similarity_matrix, image_names, dendrogram_path)
    plot_similarity_network(similarity_matrix, image_names, threshold=0.1, output_path=network_path)
    
    # Create index.html
    index_path = os.path.join(html_dir, "index.html")
    
    with open(index_path, "w") as f:
        f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Newspaper Image Clustering Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; max-width: 1200px; margin: 0 auto; }}
        h1, h2, h3 {{ color: #333; }}
        .section {{ margin-bottom: 40px; }}
        .cluster {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .cluster-title {{ display: flex; justify-content: space-between; }}
        .cluster-cohesion {{ color: #666; }}
        .images {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        .image-item {{ text-align: center; width: 200px; }}
        .image-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        .visualization {{ margin-top: 20px; text-align: center; }}
        .visualization img {{ max-width: 100%; border: 1px solid #ddd; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .highlight {{ background-color: #fffacd; }}
        .stats {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Newspaper Image Clustering Results</h1>
    <div class="section">
        <p>This report shows clustering of newspaper images based on semantic similarity of their regions, weighted by region size.</p>
        <p>Number of newspapers analyzed: {len(image_names)}</p>
        <p>Number of clusters: {clustering_results["n_clusters"]}</p>
        <p>Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <div class="stats">
        <h2>Similarity Statistics</h2>
        <p>Non-zero similarity pairs: {np.sum(similarity_matrix > 0.01) - len(image_names)}</p>
        <p>Average non-zero similarity: {np.mean(similarity_matrix[similarity_matrix > 0.01]):.4f}</p>
        <p>Max similarity between different images: {np.max(similarity_matrix - np.diag(np.diag(similarity_matrix))):.4f}</p>
    </div>
    
    <div class="section">
        <h2>Visualizations</h2>
        
        <h3>Similarity Heatmap</h3>
        <div class="visualization">
            <img src="images/similarity_heatmap.png" alt="Similarity Heatmap">
            <p>Heatmap showing pairwise similarities between newspaper images, based on weighted region comparisons.</p>
        </div>
        
        <h3>Hierarchical Clustering Dendrogram</h3>
        <div class="visualization">
            <img src="images/clustering_dendrogram.png" alt="Clustering Dendrogram">
            <p>Dendrogram showing hierarchical clustering of newspapers. Newspapers that are more similar appear closer together.</p>
        </div>
        
        <h3>Similarity Network</h3>
        <div class="visualization">
            <img src="images/similarity_network.png" alt="Similarity Network">
            <p>Network graph showing relationships between newspapers. Connected newspapers have significant region similarity.</p>
        </div>
    </div>
    
    <div class="section">
        <h2>Clusters</h2>
""")
        
        # Add each cluster
        clusters = clustering_results["clusters"]
        cohesion = clustering_results["cluster_cohesion"]
        
        # Sort clusters by cohesion (highest first)
        sorted_clusters = sorted(clusters.keys(), key=lambda x: cohesion.get(x, 0), reverse=True)
        
        for cluster_id in sorted_clusters:
            newspapers = clusters[cluster_id]
            cluster_cohesion = cohesion.get(cluster_id, 0)
            
            f.write(f"""
        <div class="cluster">
            <div class="cluster-title">
                <h3>Cluster {cluster_id}</h3>
                <span class="cluster-cohesion">Cohesion: {cluster_cohesion:.3f}</span>
            </div>
            <p>Contains {len(newspapers)} newspapers.</p>
            <table>
                <tr>
                    <th>Newspaper</th>
                </tr>
""")
            
            for newspaper in newspapers:
                f.write(f"""
                <tr>
                    <td>{newspaper}</td>
                </tr>
""")
                
            f.write("""
            </table>
        </div>
""")
        
        # Add similarity matrix table
        f.write("""
    </div>
    
    <div class="section">
        <h2>Top Similarities</h2>
        <table>
            <tr>
                <th>Newspaper 1</th>
                <th>Newspaper 2</th>
                <th>Similarity</th>
            </tr>
""")
        
        # Get all non-diagonal pairs, sorted by similarity
        pairs = []
        for i in range(len(image_names)):
            for j in range(i+1, len(image_names)):
                if similarity_matrix[i, j] > 0:
                    pairs.append((image_names[i], image_names[j], similarity_matrix[i, j]))
        
        # Sort by similarity (descending)
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Show top 50 pairs or all if less
        top_pairs = pairs[:min(50, len(pairs))]
        
        for paper1, paper2, sim in top_pairs:
            highlight = " class='highlight'" if sim > 0.5 else ""
            f.write(f"""
            <tr{highlight}>
                <td>{paper1}</td>
                <td>{paper2}</td>
                <td>{sim:.4f}</td>
            </tr>
""")
        
        if not top_pairs:
            f.write("""
            <tr>
                <td colspan="3">No similarities found between different newspapers</td>
            </tr>
""")
        
        f.write("""
        </table>
    </div>
    
    <div class="section">
        <h2>Similarity Matrix</h2>
        <table>
            <tr>
                <th>Newspaper</th>
""")
        
        # Column headers
        for name in image_names:
            short_name = name[:15] + "..." if len(name) > 18 else name
            f.write(f"<th>{short_name}</th>")
        
        f.write("</tr>")
        
        # Table data
        for i, row_name in enumerate(image_names):
            short_name = row_name[:15] + "..." if len(row_name) > 18 else row_name
            f.write(f"<tr><td>{short_name}</td>")
            
            for j in range(len(image_names)):
                similarity = similarity_matrix[i, j]
                # Color cell based on similarity (darker = more similar)
                # Use a different color for diagonal (self-similarity)
                if i == j:
                    bg_color = "#e6e6e6"  # Light gray for diagonal
                else:
                    bg_color = f"rgba(0, 100, 255, {similarity:.2f})"
                f.write(f'<td style="background-color: {bg_color};">{similarity:.3f}</td>')
            
            f.write("</tr>")
        
        f.write("""
        </table>
    </div>
    
</body>
</html>
""")
    
    logger.info(f"Created HTML report at {index_path}")
    return index_path

def main():
    """
    Main function that coordinates the newspaper image clustering workflow.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Newspaper Image Clustering')
    parser.add_argument('--n-clusters', type=int, default=None, help='Number of clusters to form (default: automatic)')
    parser.add_argument('--similarity-threshold', type=float, default=0.1, 
                        help='Minimum similarity score to consider regions as related (default: 0.1)')
    parser.add_argument('--no-html', action='store_true', help='Skip HTML report generation')
    parser.add_argument('--reset', action='store_true', help='Reset progress and start fresh')
    parser.add_argument('--diagnostic', action='store_true', help='Run in diagnostic mode with detailed logging')
    parser.add_argument('--include-same-prefix', action='store_true', 
                        help='Include comparisons between files with identical prefixes (default: exclude)')
    parser.add_argument('--prefix-length', type=int, default=20,
                        help='Number of characters to use for prefix matching (default: 20)')
    
    args = parser.parse_args()
    
    # Set up detailed logging if diagnostic mode is enabled
    if args.diagnostic:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.info("Running in diagnostic mode with detailed logging")
    
    n_clusters = args.n_clusters
    similarity_threshold = args.similarity_threshold
    generate_html = not args.no_html
    reset_progress = args.reset
    skip_same_prefix = not args.include_same_prefix
    prefix_length = args.prefix_length
    
    logger.info("Starting newspaper image weighted clustering")
    logger.info(f"Skip same prefix: {skip_same_prefix} (prefix length: {prefix_length})")
    
    # Create necessary directories
    os.makedirs(WEIGHTED_CLUSTERING_FOLDER, exist_ok=True)
    
    # Reset progress if requested
    if reset_progress:
        logger.info("Resetting clustering progress as requested")
        if os.path.exists(WEIGHTED_CLUSTERING_PROGRESS_FILE):
            os.remove(WEIGHTED_CLUSTERING_PROGRESS_FILE)
        save_clustering_progress({"completed_comparisons": []})
    
    # Initialize database connection
    chroma_client, collection = initialize_db()
    
    # Get image paths
    image_paths = get_image_paths(IMAGE_FOLDER)
    
    if not image_paths:
        logger.error(f"No image files found in {IMAGE_FOLDER}. Exiting.")
        return
    
    logger.info(f"Found {len(image_paths)} image files in {IMAGE_FOLDER}")
    
    # Compute image similarity matrix
    similarity_matrix, image_names = compute_image_similarity_matrix(
        collection, 
        image_paths,
        similarity_threshold=similarity_threshold,
        skip_same_prefix=skip_same_prefix,
        prefix_length=prefix_length
    )
    
    if similarity_matrix is None or similarity_matrix.size == 0 or image_names is None or len(image_names) == 0:
        logger.error("Failed to compute valid similarity matrix or image names. Exiting.")
        return
    
    # Save similarity matrix for future use
    similarity_matrix_path = os.path.join(WEIGHTED_CLUSTERING_FOLDER, "similarity_matrix.npy")
    np.save(similarity_matrix_path, similarity_matrix)
    
    image_names_path = os.path.join(WEIGHTED_CLUSTERING_FOLDER, "image_names.json")
    with open(image_names_path, 'w') as f:
        json.dump(image_names, f)
    
    logger.info(f"Saved similarity matrix to {similarity_matrix_path}")
    
    # Cluster images
    clustering_results = cluster_images(similarity_matrix, image_names, n_clusters=n_clusters)
    
    if clustering_results is None:
        logger.error("Clustering failed. Exiting.")
        return
    
    # Save clustering results
    clustering_path = os.path.join(WEIGHTED_CLUSTERING_FOLDER, "clustering_results.json")
    with open(clustering_path, 'w') as f:
        json.dump(clustering_results, f, indent=2)
    
    logger.info(f"Saved clustering results to {clustering_path}")
    
    # Create visualizations
    plots_dir = os.path.join(WEIGHTED_CLUSTERING_FOLDER, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate visualizations
    plot_similarity_heatmap(
        similarity_matrix, 
        image_names, 
        os.path.join(plots_dir, "similarity_heatmap.png")
    )
    
    plot_dendrogram(
        similarity_matrix, 
        image_names, 
        os.path.join(plots_dir, "clustering_dendrogram.png")
    )
    
    plot_similarity_network(
        similarity_matrix, 
        image_names, 
        threshold=0.1,  # Lower threshold to show more connections
        output_path=os.path.join(plots_dir, "similarity_network.png")
    )
    
    # Create HTML report if requested
    if generate_html:
        html_dir = os.path.join(WEIGHTED_CLUSTERING_FOLDER, "html_report")
        report_path = create_html_report(
            similarity_matrix, 
            image_names, 
            clustering_results, 
            html_dir
        )
        logger.info(f"HTML report created at {report_path}")
    
    logger.info("Newspaper image clustering completed successfully!")

if __name__ == "__main__":
    main()