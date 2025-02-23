import click
from .kmer_count import preproc
from .motif_discovery import  scan_motif, ex_hamball, draw_logo
from .visualization import visualize_kmers

@click.group()
def cli():
    """
    KMAP: visualize kmers in 2d.
    """
    pass

def display_paper_info():
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
