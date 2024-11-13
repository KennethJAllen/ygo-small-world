"""Streamlit app."""

import streamlit as st
from ygo_small_world import small_world_bridge_generator as sw
from ygo_small_world import graph_adjacency_visualizer as gav

st.title('Yu-Gi-Oh! Small World')

data_load_state = st.text('Loading data...')

deck_monster_names = ['Ash Blossom & Joyous Spring',
                        'Effect Veiler',
                        'Ghost Belle & Haunted Mansion',
                        'Mathmech Addition',
                        'Mathmech Circular',
                        'Mathmech Diameter',
                        'Mathmech Multiplication',
                        'Mathmech Sigma',
                        'Mathmech Subtraction',
                        'Lava Golem',
                        'Nibiru, the Primal Being',
                        'Parallel eXceed']

#cards that are required to connect with a bridge
required_target_names = ['Mathmech Circular']

bridges = sw.find_best_bridges(deck_monster_names, required_target_names, top=20)
bridges

ydk_file = 'examples/sample_deck.ydk'
graph = gav.ydk_to_graph(ydk_file)
gav.plot_graph(graph)

data_load_state.text('Loading data...done!')