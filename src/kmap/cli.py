import click
from .kmer_count import preproc
from .motif_discovery import  scan_motif, ex_hamball, draw_logo
from .visualization import visualize_kmers
from .util import align_conseq, extract_motif_locations, plot_cooccurrence_network
import importlib.metadata
from pathlib import Path

@click.group()
def cli():
    """
    KMAP: visualize kmers in 2d.
    """
    pass


def display_paper_info():
    print()
    print(f"KMAP version: {importlib.metadata.version('kmer-map')}")
    print()
    print("Citation")
    print("KMAP: Kmer Manifold Approximation and Projection for visualizing DNA sequences")
    print("Chengbo Fu, Einari A. Niskanen, Gong-Hong Wei, Zhirong Yang, Marta Sanvicente-García, Marc Güell, Lu Cheng*")
    print("BioRxiv")
    print("2024")
    print("DOI: https://doi.org/10.1101/2024.04.12.589197")


cli.add_command(preproc)
cli.add_command(scan_motif)
cli.add_command(ex_hamball)
cli.add_command(draw_logo)
cli.add_command(visualize_kmers)
cli.add_command(align_conseq)
cli.add_command(extract_motif_locations)

# The plot_network command has been removed

@cli.command()
@click.option('--res_dir', default='./test/', help='Path to result directory')
@click.option('--cutoff', default=0.7, help='Co-occurrence frequency cutoff')
@click.option('--output-file', default='cooccurrence_network.png', help='Output file name for the network plot')
def plot_network(res_dir, cutoff, output_file):
    """Plot co-occurrence network from matrix files."""
    co_occur_file = Path(res_dir) / "co_occurence/co_occurence_mat.norm.tsv"
    dist_file = Path(res_dir) / "co_occurence/co_occurence_motif_dist_mat.tsv"
    plot_cooccurrence_network(co_occur_file, dist_file, co_occur_cutoff=cutoff, output_file=output_file)
