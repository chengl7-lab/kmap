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
    print("This software is affiliated with the following paper:")
    print("Title: Your Paper Title")
    print("Authors: First Author, Second Author, Third Author")
    print("Journal: Journal Name")
    print("Year: 2023")
    print("DOI: https://doi.org/your-paper-doi")


cli.add_command(preproc)
cli.add_command(scan_motif)
cli.add_command(ex_hamball)
cli.add_command(draw_logo)
cli.add_command(visualize_kmers)
