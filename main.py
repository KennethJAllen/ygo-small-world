import argparse
from ygo_small_world import small_world_bridge_generator as sw
from ygo_small_world import graph_adjacency_visualizer as gav
from ygo_small_world.update_data import update_card_data

def main():
    """
    Manages command-line interface for analyzing Yu-Gi-Oh! Small World connections from a .ydk file.
    
    Args:
    - decklist (str): Path to a text file with one card name per line.
    - action (str): Type of analysis ('bridges', 'matrix', 'graph', 'matrix_image').
    
    Optional Args:
    - --update_data (bool): Updates card data cardinfo.json.
    - --top (int): Limits the number of bridges displayed when action is 'bridges'. Defaults to showing 20.
    - --square (bool): Squares the adjacency matrix if set for 'matrix' or 'matrix_image' actions. Default is False.
    
    Examples:
    py main.py "deck.ydk" bridges --top 10
    py main.py "deck.ydk" matrix
    py main.py "deck.ydk" graph --update_data
    py main.py "deck.ydk" matrix_image --squared
    """
    parser = argparse.ArgumentParser(description="Process a Yu-Gi-Oh! deck to analyze Small World connections.")
    parser.add_argument('ydk_file', type=str, help='File path to the ydk (Yu-Gi-Oh! deck) file.')
    parser.add_argument('action', type=str, choices=['bridges', 'matrix', 'graph', 'matrix_image'],
                        help='Action to perform: bridges, matrix, graph, or matrix_image')

    # optional arguments
    parser.add_argument('--update_data', type=bool, nargs='?', const=True, default=False,
                        help='Update card data.')
    parser.add_argument('--top', type=int, default=20,
                        help='Number of top bridges to display.')
    parser.add_argument('--squared', type=bool, nargs='?', const=True, default=False,
                        help='Whether to square the adjacency matrix.')

    args = parser.parse_args()

    # update card data
    if args.update_data:
        update_card_data()

    # Process based on the action
    if args.action == 'bridges':
        bridges = sw.find_best_bridges_from_ydk(args.ydk_file, top=args.top)
        print(bridges)
    elif args.action == 'matrix':
        matrix = sw.ydk_to_labeled_adjacency_matrix(args.ydk_file, squared=args.squared)
        print(matrix)
    elif args.action == 'graph':
        graph = gav.ydk_to_graph(args.ydk_file)
        gav.plot_graph(graph)
    elif args.action == 'matrix_image':
        matrix_image = gav.ydk_to_matrix_image(args.ydk_file, squared=args.squared)
        gav.plot_matrix(matrix_image)

if __name__ == "__main__":
    main()

