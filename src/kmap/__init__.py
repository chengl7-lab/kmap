from .cli import cli, display_paper_info
from .kmer_count import FileNameDict

def main() -> object:
    display_paper_info()
    cli(prog_name="kmap")
