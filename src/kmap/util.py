import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import pandas as pd
from Bio import Align
from pathlib import Path
import matplotlib.colors  # Added import for rgb2hex function
import click
from operator import itemgetter
import networkx as nx
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import re


# Main process
@click.command(name="align_conseq")
@click.option(
    '--conseq_csv_file',
    type=str,
    help='Input conseq csv file',
    required=True
    )
@click.option(
    '--out_dir',
    type=str,
    default="./conseq_similarity",
    help='Result directory for storing conseq similarities',
    required=False
    )
def align_conseq(conseq_csv_file: str, out_dir: str = "./conseq_similarity"):
    _align_conseq(conseq_csv_file, out_dir)


@click.command(name="extract_motif_locations")
@click.option(
    '--bed_file',
    type=str,
    help='Input bed file with no header, each line corresponds to each read in the input fasta file',
    required=True
    )
@click.option(
    '--conseq_file',
    type=str,
    help='Input conseq file',
    default="./final_conseq.txt",
    required=False
)
@click.option(
    '--motif_occurrence_file',
    type=str,
    help='Input motif occurrence file',
    default="./final.motif_occurence.csv",
    required=False
)
@click.option(
    '--output_dir',
    type=str,
    default="./motif_locations",
    help='Output directory for storing motif locations',
    required=False
)
def extract_motif_locations(bed_file: str, conseq_file: str, motif_occurrence_file: str, output_dir: str):
    _extract_motif_locations(bed_file, conseq_file, motif_occurrence_file, output_dir)


# 1. Read and process the CSV file
def read_and_process_csv(file_path):
    df = pd.read_csv(file_path)
    motifs = df.iloc[:, 2:4].values.tolist()
    return [(i, seq, rc) for i, (seq, rc) in enumerate(motifs)]


# 2. Filter out repetitive sequences
def remove_redundant_sequences(motifs):
    def is_repetitive(seq):
        if len(set(seq)) == 1:
            return True
        for i in range(1, len(seq) // 2 + 1):
            if len(seq) % i == 0 and seq == seq[:i] * (len(seq) // i):
                return True
        return False

    return [motif for motif in motifs if not is_repetitive(motif[1])]


# 3. Create new consensus sequences
def create_new_consensus_sequences(motifs):
    new_motifs = []
    for id, seq, rc in motifs:
        new_motifs.append(f"m{id}-FS-{seq}")
        new_motifs.append(f"m{id}-RC-{rc}")
    return new_motifs


# 4. Compute normalized score
def compute_normalized_similarity_score(seq1, seq2):
    # Extract the sequence part
    seq1 = seq1.split('-')[-1]
    seq2 = seq2.split('-')[-1]

    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -1

    alignments = aligner.align(seq1, seq2)
    if len(alignments) == 0:
        return 0

    best_alignment = max(alignments, key=lambda x: x.score)
    alignment_length = best_alignment.aligned[0][-1][1] - best_alignment.aligned[0][0][0]
    normalized_score = alignment_length / min(len(seq1), len(seq2))
    return normalized_score


# 5. Build score matrix
def build_score_matrix(motifs):
    n = len(motifs)
    score_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            score = compute_normalized_similarity_score(motifs[i], motifs[j])
            distance = 1 - score  # Convert similarity to distance
            score_matrix[i, j] = distance
            score_matrix[j, i] = distance
    return score_matrix


# 6. Hierarchical clustering with pairwise alignments
def hierarchical_clustering_with_alignments(score_matrix, new_motifs, output_dir="conseq_similarity"):
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    Z = linkage(score_matrix, 'average')

    # Calculate the cutoff distance
    max_dist = max(Z[:, 2])
    cutoff_dist = 0.5 * max_dist

    # Automatically determine clusters
    clusters = fcluster(Z, cutoff_dist, criterion='distance')
    num_clusters = len(set(clusters))

    # Create a color map for clusters using discrete colors
    cmap = plt.colormaps[
        'tab20']  # You can change 'tab20' to other discrete colormaps like 'Set1', 'Set2', 'Set3', etc.
    n_colors = 20
    colors = [cmap(i) for i in np.linspace(0, 1, n_colors)]

    # Function to get color for each cluster
    def get_color(i):
        return colors[i % n_colors]

    cluster_colors = [get_color(i) for i in range(num_clusters)]

    # Create a mapping of cluster numbers to color indices
    cluster_to_color = {c: i for i, c in enumerate(sorted(set(clusters), reverse=True))}

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(14, len(new_motifs) * 0.3))  # Increased width to accommodate legend

    def get_cluster_for_node(node, n_leaves):
        if node < n_leaves:
            return clusters[node]
        else:
            left = int(Z[node - n_leaves, 0])
            right = int(Z[node - n_leaves, 1])
            left_cluster = get_cluster_for_node(left, n_leaves)
            right_cluster = get_cluster_for_node(right, n_leaves)
            return left_cluster if left_cluster == right_cluster else -1

    def link_color_func(k):
        cluster = get_cluster_for_node(k, len(new_motifs))
        if cluster != -1:
            color = cluster_colors[cluster_to_color[cluster]]
            return matplotlib.colors.rgb2hex(color[:3])
        return 'grey'  # Color for branches above the cutoff

    ddata = dendrogram(
        Z,
        labels=new_motifs,
        orientation='left',
        leaf_font_size=8,
        color_threshold=cutoff_dist,
        ax=ax,
        link_color_func=link_color_func
    )

    ax.set_title(f'Hierarchical Clustering (Number of clusters: {num_clusters})')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Motifs')

    # Move y-axis to the right side
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    # Color the labels and add cluster numbers
    for label in ax.get_yticklabels():
        cluster_num = clusters[new_motifs.index(label.get_text())]
        label.set_color(cluster_colors[cluster_to_color[cluster_num]])
        label.set_text(f"cluster_{cluster_num} | {label.get_text()}")

    # Add legend for cluster colors in reverse order
    legend_elements = [plt.Line2D([0], [0], color=cluster_colors[cluster_to_color[c]], lw=4, label=f'Cluster {c}')
                       for c in sorted(set(clusters), reverse=True)]
    ax.legend(handles=legend_elements, title="Clusters", loc='upper left', bbox_to_anchor=(0, 1))

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path / 'dendrogram.pdf', bbox_inches='tight')
    plt.close()

    # Perform pairwise alignments for each cluster
    for i in range(1, num_clusters + 1):  # Change this line
        cluster_motifs = [(new_motifs[j], j) for j in range(len(new_motifs)) if clusters[j] == i]

        if len(cluster_motifs) > 1:  # Only create alignment if there's more than one sequence
            # Perform pairwise alignments
            pairwise_alignments = perform_pairwise_alignments(cluster_motifs)

            # Write alignments to a file
            with open(output_path / f'cluster_{i}_pairwise_alignments.txt', 'w') as f:
                for seq1, seq2, alignment in pairwise_alignments:
                    f.write(f"Alignment between {seq1} and {seq2}:\n")
                    f.write(str(alignment) + "\n")
                    f.write(f"Score: {alignment.score}\n")
                    f.write("\n")

    print(f"Clustering complete. Results saved in {output_path}")


def perform_pairwise_alignments(sequences):
    alignments = []
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -1

    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            seq1, idx1 = sequences[i]
            seq2, idx2 = sequences[j]
            alignment = aligner.align(seq1.split('-')[-1], seq2.split('-')[-1])[0]
            alignments.append((seq1, seq2, alignment))
    return alignments


def _align_conseq(conseq_csv_file: str, out_dir: str = "./conseq_similarity"):
    # 1. Read and process the CSV file
    motifs = read_and_process_csv(Path(conseq_csv_file))

    # 2. Filter out repetitive sequences
    filtered_motifs = remove_redundant_sequences(motifs)

    if len(filtered_motifs) < 2:
        print("Less than 2 motifs after filtering repetitive conseqs. Quit!")
        return

    # 3. Create new consensus sequences
    new_motifs = create_new_consensus_sequences(filtered_motifs)

    # 4 & 5. Compute normalized scores and build score matrix
    score_matrix = build_score_matrix(new_motifs)

    # 6. Perform hierarchical clustering and pairwise alignments
    hierarchical_clustering_with_alignments(score_matrix, new_motifs, output_dir=out_dir)

def merge_intervals(intervals):
    sorted_intervals = sorted(intervals, key=itemgetter(0))
    merged = []
    for start, end in sorted_intervals:
        if not merged or merged[-1][1] < start:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def _extract_motif_locations(bed_file: str, conseq_file: str, motif_occurrence_file: str, output_dir: str):
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read input files
    bed_df = pd.read_csv(bed_file, sep='\t', header=None)
    if len(bed_df.columns) == 3:
        bed_df.columns = ['chrom', 'start', 'end']
    elif len(bed_df.columns) == 6:
        bed_df.columns = ['chrom', 'start', 'end', 'name', 'score', 'strand']
    else:
        raise ValueError("Input BED file should have either 3 or 6 columns")
    conseq_list = Path(conseq_file).read_text().splitlines()
    motif_occurrences = pd.read_csv(motif_occurrence_file, sep=';', index_col=0)

    # Process each consensus sequence
    for i, conseq in enumerate(conseq_list):
        motif_bed = []
        
        # Iterate through motif occurrences
        for read_index, occurrences in motif_occurrences.iterrows():
            read_bed = bed_df.iloc[read_index]

            # Use iloc to access the i-th column
            if pd.isna(occurrences.iloc[i]):
                continue

            windows = []
            # Process each occurrence for the current consensus sequence
            for occurrence in occurrences.iloc[i].split(","):
                rel_start = int(occurrence)

                # Translate relative position to genomic location
                abs_start = read_bed['start'] + rel_start
                abs_end = abs_start + len(conseq)
                windows.append([abs_start, abs_end])

            # Merge overlapping windows
            merged_windows = merge_intervals(windows)

            # Store the merged windows
            for abs_start, abs_end in merged_windows:
                # Create BED entry
                motif_bed.append([
                    read_bed['chrom'],
                    abs_start,
                    abs_end,
                    f"motif_{i}_{read_index}",
                    0,  # score (you can modify this if needed)
                    read_bed['strand']
                ])
                
        motif_bed.sort()
        # Write BED file for the current consensus sequence
        output_file = output_path / f"motif_{i}_{conseq}_locations.bed"
        pd.DataFrame(motif_bed, columns=['chrom', 'start', 'end', 'name', 'score', 'strand']).to_csv(
            output_file, sep='\t', header=True, index=False
        )

    print(f"Motif location extraction complete. Results saved in {output_path}")


def plot_cooccurrence_network(co_occur_file, dist_file, co_occur_cutoff=0.7, output_file='cooccurrence_network.pdf'):
    # Read the TSV files
    df_co_occur = pd.read_csv(co_occur_file, sep='\t', index_col=0)
    df_dist = pd.read_csv(dist_file, sep='\t', index_col=0)

    # Create a graph
    G = nx.Graph()

    # Add nodes using column names
    for node in df_co_occur.columns:
        G.add_node(node)

    # Add edges using the upper triangle of the matrix
    for i, col in enumerate(df_co_occur.columns):
        for j, row in enumerate(df_co_occur.columns[i+1:], start=i+1):
            co_occur_value = df_co_occur.iloc[i, j]
            dist_value = df_dist.iloc[i, j]
            if co_occur_value > co_occur_cutoff:
                G.add_edge(col, row, weight=co_occur_value, distance=dist_value)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Calculate node sizes based on degree
    node_sizes = [300 * (1 + G.degree(node)) for node in G.nodes()]

    # Draw the network
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Adjust positions to bring disconnected nodes closer
    max_x = max(coord[0] for coord in pos.values())
    max_y = max(coord[1] for coord in pos.values())
    for node, coords in pos.items():
        if G.degree(node) == 0:
            pos[node] = (0.5 * max_x * (0.8 + 0.4 * np.random.random()),
                         0.5 * max_y * (0.8 + 0.4 * np.random.random()))

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    # Draw edges with uniform width but varying colors
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    uniform_width = 2  # You can adjust this value to change the edge thickness
    
    # Create a colormap
    cmap = plt.cm.viridis
    norm = Normalize(vmin=co_occur_cutoff, vmax=max(edge_weights) if edge_weights else 1)
    
    # Draw edges with color mapping
    if G.edges():
        edges = nx.draw_networkx_edges(G, pos, width=uniform_width, edge_color=edge_weights, 
                                       edge_cmap=cmap, edge_vmin=co_occur_cutoff, 
                                       edge_vmax=max(edge_weights), ax=ax)

        # Add edge labels (distances)
        edge_labels = nx.get_edge_attributes(G, 'distance')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

        # Add colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Co-occurrence Frequency', 
                            orientation='horizontal', pad=0.08, aspect=30)

    plt.title(f"Co-occurrence Network (cutoff: {co_occur_cutoff})")
    ax.axis('off')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Network plot saved as {output_file}")


if __name__ == "__main__":
    conseq_csv_file = 'test/final_conseq.info.csv'
    # conseq_csv_file = 'test/candidate_conseq.csv'
    _align_conseq(conseq_csv_file, out_dir = "./conseq_similarity")

    bed_file = "./c2.bed"
    conseq_file = "./test/final_conseq.txt"
    motif_occurrence_file = "./test/final.motif_occurence.csv"
    out_dir = "./test/conseq_similarity"
    _extract_motif_locations(bed_file, conseq_file, motif_occurrence_file, out_dir)