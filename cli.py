"""Small World command line interface."""
from pathlib import Path
import argparse
from matplotlib import pyplot as plt
from ygo_small_world.small_world_bridge_generator import AllCards, Deck, Bridges
from ygo_small_world import graph_adjacency_visualizer as gav
from ygo_small_world.update_data import update_card_data

def cli():
    """
    Manages command-line interface for analyzing Yu-Gi-Oh! Small World connections from a .ydk file.
    Saves bridges, adjacency matrix plot, squared adjacency matrix plot, and graph plot to output path.
    Default output path is ./outputs
    
    Args:
    - decklist (str): Path to a text file with one card name per line.
    
    Optional Args:
    - --update_data (bool): Updates card data cardinfo.json.
    - --output (Path): Save to specified path.
    
    Example:
    py -m cli "data/sample_deck.ydk"
    """
    parser = argparse.ArgumentParser(description="Process a .ydk (Yu-Gi-Oh! deck) file to analyze Small World connections.")
    parser.add_argument('ydk_file', type=str, help='File path to the ydk (Yu-Gi-Oh! deck) file.')

    # optional arguments
    parser.add_argument('--update_data', type=bool, nargs='?', const=True, default=False, help='Update card data.')
    parser.add_argument('--output', type=str, help="Output path.", default=None)

    args = parser.parse_args()

    # update card data
    if args.update_data:
        update_card_data()

    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent / 'output'
    output_path.mkdir(exist_ok=True, parents=True)

    all_cards = AllCards()
    deck = Deck(all_cards, ydk_path=args.ydk_file)
    bridges = Bridges(deck, all_cards)

    # Save bridges to CSV
    bridges.get_bridge_df().to_csv(output_path / "bridges.csv")

    # Save graph
    graph = gav.graph_fig(deck, save_path = output_path / "graph.png")
    plt.close(graph)
    
    # Save adjacency matrices
    adjacency_fig = gav.matrix_fig(deck, save_path = output_path / "adjacency_matrix.png")
    plt.close(adjacency_fig)

    squared_adjacency_fig = gav.matrix_fig(deck, squared=True, save_path = output_path / "squared_adjacency_matrix.png")
    plt.close(squared_adjacency_fig)

if __name__ == "__main__":
    cli()
