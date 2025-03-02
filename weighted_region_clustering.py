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

def compute_image_similarity_matrix(collection, image_paths, similarity_threshold=REGION_SIMILARITY_THRESHOLD):
    """
    Compute a similarity matrix between all pairs of newspaper images based on
    the similarity of their sub-regions, weighted by region area.
    
    Args:
        collection: ChromaDB collection containing region embeddings
        image_paths: List of paths to newspaper images
        similarity_threshold: Minimum similarity score to consider two regions as related
        
    Returns:
        numpy.ndarray: Similarity matrix between all pairs of images
        list: Image names corresponding to matrix rows/columns
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
        where={"is_region": {"$eq": True}}  # Only include regions
    )
    
    if not all_entries or "ids" not in all_entries or not all_entries["ids"]:
        logger.warning("No regions found in the database. Make sure regions have been processed first.")
        return None, None
    
    # Group regions by parent image
    image_to_regions = defaultdict(list)
    region_embeddings = {}
    region_areas = {}
    
    for i, region_id in enumerate(all_entries["ids"]):
        metadata = all_entries["metadatas"][i]
        embedding = all_entries["embeddings"][i]
        
        if metadata and embedding:
            parent_image = metadata.get("parent_image_name")
            area_percentage = metadata.get("area_percentage", 0)
            region_type = metadata.get("region_type")
            
            if parent_image and area_percentage > 0 and region_type in REGION_TYPES_TO_PROCESS:
                image_to_regions[parent_image].append(region_id)
                region_embeddings[region_id] = embedding
                region_areas[region_id] = area_percentage / 100.0  # Convert to fraction
    
    # Compute cross-image region similarities
    logger.info("Computing cross-image region similarities...")
    
    # For each pair of images
    total_pairs = (n_images * (n_images - 1)) // 2
    
    with tqdm(total=total_pairs, desc="Computing image similarities") as pbar:
        for i in range(n_images):
            img_i = image_names[i]
            regions_i = image_to_regions.get(img_i, [])
            
            if not regions_i:
                continue
                
            for j in range(i+1, n_images):
                img_j = image_names[j]
                regions_j = image_to_regions.get(img_j, [])
                
                if not regions_j:
                    continue
                
                # Check if this pair has already been processed
                pair_key = f"{img_i}_{img_j}"
                if is_clustering_completed(pair_key):
                    pbar.update(1)
                    continue
                
                # For each region in image i, find its similarity to all regions in image j
                weighted_similarities = []
                
                for region_i in regions_i:
                    embedding_i = region_embeddings.get(region_i)
                    area_i = region_areas.get(region_i, 0)
                    
                    if embedding_i is None or area_i == 0:
                        continue
                    
                    # Query for similar regions from image j
                    results = collection.query(
                        query_embeddings=[embedding_i],
                        n_results=len(regions_j),
                        include=["metadatas", "distances"],
                        where={"parent_image_name": {"$eq": img_j}}
                    )
                    
                    if not results or "ids" not in results or not results["ids"][0]:
                        continue
                    
                    # Calculate weighted similarities
                    for k, (region_j, distance) in enumerate(zip(results["ids"][0], results["distances"][0])):
                        metadata_j = results["metadatas"][0][k]
                        area_j = metadata_j.get("area_percentage", 0) / 100.0
                        
                        if distance >= similarity_threshold and area_j > 0:
                            # Weight by product of areas (geometric mean)
                            # This weights more heavily relationships between large regions
                            weighted_similarity = (1.0 - distance) * area_i * area_j
                            weighted_similarities.append(weighted_similarity)
                
                # Compute overall weighted similarity between the two images
                if weighted_similarities:
                    # Sum of weighted similarities (higher is more similar)
                    similarity_score = np.sum(weighted_similarities)
                    
                    # Store in the similarity matrix
                    similarity_matrix[i, j] = similarity_score
                    similarity_matrix[j, i] = similarity_score  # Symmetrical
                
                # Mark pair as completed
                mark_clustering_as_completed(pair_key)
                pbar.update(1)
    
    # Normalize similarity matrix
    if np.max(similarity_matrix) > 0:
        similarity_matrix = similarity_matrix / np.max(similarity_matrix)
    
    return similarity_matrix, image_names

def cluster_images(similarity_matrix, image_names, n_clusters=None):
    """
    Cluster newspaper images based on their similarity matrix.
    
    Args:
        similarity_matrix: Similarity matrix between all pairs of images
        image_names: List of image names corresponding to matrix rows/columns
        n_clusters: Number of clusters to form (if None, determined automatically)
        
    Returns:
        dict: Clustering results
    """
    # Convert similarity to distance (1 - similarity)
    distance_matrix = 1.0 - similarity_matrix
    
    # Apply hierarchical clustering
    logger.info("Applying hierarchical clustering...")
    
    # If n_clusters is None, determine it automatically based on the distance matrix
    if n_clusters is None:
        # Use the elbow method or silhouette score to determine optimal number of clusters
        from sklearn.metrics import silhouette_score
        
        max_clusters = min(10, len(image_names))
        best_score = -1
        best_n_clusters = 2  # Default to at least 2 clusters
        
        for k in range(2, max_clusters + 1):
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=k,
                    affinity='precomputed',
                    linkage='average'
                ).fit(distance_matrix)
                
                if len(np.unique(clustering.labels_)) > 1:  # Only calculate if there are at least 2 clusters
                    score = silhouette_score(distance_matrix, clustering.labels_, metric='precomputed')
                    
                    if score > best_score:
                        best_score = score
                        best_n_clusters = k
            except Exception as e:
                logger.warning(f"Error computing silhouette score for k={k}: {str(e)}")
        
        n_clusters = best_n_clusters
        logger.info(f"Automatically determined optimal number of clusters: {n_clusters}")
    
    # Perform final clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
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
            # Extract the submatrix for this cluster
            submatrix = similarity_matrix[np.ix_(indices, indices)]
            # Average similarity excluding self-similarity
            mask = ~np.eye(submatrix.shape[0], dtype=bool)
            cohesion = np.mean(submatrix[mask]) if np.any(mask) else 0
            cluster_cohesion[label] = float(cohesion)
        else:
            cluster_cohesion[label] = 0.0
    
    return {
        "n_clusters": n_clusters,
        "clusters": {str(k): v for k, v in clusters.items()},
        "cluster_cohesion": cluster_cohesion,
        "labels": clustering.labels_.tolist()
    }

def plot_similarity_heatmap(similarity_matrix, image_names, output_path=None):
    """
    Plot a heatmap of the image similarity matrix.
    
    Args:
        similarity_matrix: Similarity matrix between all pairs of images
        image_names: List of image names corresponding to matrix rows/columns
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    plt.imshow(similarity_matrix, cmap="viridis", interpolation="nearest")
    
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
    plt.figure(figsize=(14, 8))
    
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
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for i, name in enumerate(image_names):
        # Trim name for display
        short_name = name[:15] + "..." if len(name) > 18 else name
        G.add_node(i, name=short_name)
    
    # Add edges
    for i in range(len(image_names)):
        for j in range(i+1, len(image_names)):
            similarity = similarity_matrix[i, j]
            if similarity > threshold:
                G.add_edge(i, j, weight=similarity)
    
    # Create the plot
    plt.figure(figsize=(12, 12))
    
    # Position nodes using force-directed layout
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.8)
    
    # Draw edges with width proportional to similarity
    edges = G.edges()
    weights = [G[u][v]["weight"] * 5 for u, v in edges]  # Scale for visibility
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, alpha=0.5)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, "name"), font_size=8)
    
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
    plot_similarity_network(similarity_matrix, image_names, threshold=0.3, output_path=network_path)
    
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
    parser.add_argument('--similarity-threshold', type=float, default=REGION_SIMILARITY_THRESHOLD, 
                        help='Minimum similarity score to consider regions as related')
    parser.add_argument('--no-html', action='store_true', help='Skip HTML report generation')
    
    args = parser.parse_args()
    
    n_clusters = args.n_clusters
    similarity_threshold = args.similarity_threshold
    generate_html = not args.no_html
    
    logger.info("Starting newspaper image weighted clustering")
    
    # Create necessary directories
    os.makedirs(WEIGHTED_CLUSTERING_FOLDER, exist_ok=True)
    
    # Initialize database connection
    chroma_client, collection = initialize_db()
    
    # Get image paths
    image_paths = get_image_paths(IMAGE_FOLDER)
    
    if not image_paths:
        logger.error(f"No image files found in {IMAGE_FOLDER}. Exiting.")
        return
    
    logger.info(f"Found {len(image_paths)} newspaper images to analyze")
    
    # Compute image similarity matrix
    similarity_matrix, image_names = compute_image_similarity_matrix(
        collection, 
        image_paths,
        similarity_threshold=similarity_threshold
    )
    
    if similarity_matrix is None or image_names is None:
        logger.error("Failed to compute similarity matrix. Exiting.")
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
        threshold=0.3,
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