"""YGO Small World Streamlit app."""
import os
from pathlib import Path
import tempfile
import streamlit as st
from ygo_small_world import bridges as sw
from ygo_small_world import plots as gav

def save_temp_file(uploaded_file):
    """Save uploaded file to temporary location and return the path"""
    # Create a temporary file
    temp_dir = tempfile.mkdtemp()
    print(uploaded_file)
    temp_path = os.path.join(temp_dir, uploaded_file.name)

    # Write the file
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return Path(temp_path)

def main():
    """Main entry point to app."""
    st.title("Yu-Gi-Oh! Small World Bridge Generator")
    st.write("""
    Upload your .ydk file to analyze Small World connections in your deck.
    You'll see bridge recommendations and visualizations of the connections between your cards.
    """)
    # File uploader
    uploaded_file = st.file_uploader("Choose your .ydk file", type=['ydk'])

    use_sample_deck = st.button("Use Sample Deck")

    file_path = None
    if uploaded_file is not None:
        file_path = save_temp_file(uploaded_file)
    elif use_sample_deck:
        file_path = Path.cwd() / 'data' / 'sample_deck.ydk'

    # Process the file when uploaded
    if file_path is not None:
        st.success(f"Using deck: {file_path.stem.replace('_', ' ').title()}")
        try:
            # 1. Small World Bridges Section
            st.header("Top Small World Bridges")
            with st.spinner("Finding bridges..."):
                try:
                    df_bridges = sw.find_best_bridges_from_ydk(file_path, top=150)
                    if df_bridges is not None and not df_bridges.empty:
                        df_bridges = df_bridges.copy()
                        df_bridges['bridge_score'] = df_bridges['bridge_score'].map(lambda x: f"{x*100:.2f}%")
                        df_bridges.columns = df_bridges.columns.str.title()
                        df_bridges.columns = df_bridges.columns.str.replace('_', ' ')
                    if df_bridges is not None and not df_bridges.empty:
                        st.dataframe(df_bridges, use_container_width=True, hide_index=True)
                    else:
                        st.warning("No viable bridges found in the deck.")
                except Exception as e:
                    st.error(f"Unexpected error finding bridges: {str(e)}")

            st.divider()

            # 2. Graph Section
            st.header("Graph Visualization")
            with st.spinner("Generating graph..."):
                try:
                    graph = gav.ydk_to_graph(file_path)
                    display_fig = gav.graph_fig(graph)
                    st.pyplot(display_fig)
                except Exception as e:
                    st.error(f"Error generating network graph: {str(e)}")

            st.divider()

            # 3. Small World Connections Section
            st.header("Small World Connections")
            with st.spinner("Generating connections..."):
                try:
                    connection_data = connection_data = gav.ydk_to_matrix_image(file_path, squared=False)
                    display_fig = gav.matrix_fig(connection_data)
                    st.pyplot(display_fig)
                except Exception as e:
                    st.error(f"Error generating matrix heatmap: {str(e)}")

            st.divider()

            # 4. Matrix Heatmap Section
            st.header("Searchable Cards Heatmap")
            with st.spinner("Generating searchable cards..."):
                try:
                    heatmap = gav.ydk_to_matrix_image(file_path, squared=True)
                    display_fig = gav.matrix_fig(heatmap)
                    st.pyplot(display_fig)
                except Exception as e:
                    st.error(f"Error generating matrix heatmap: {str(e)}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please make sure your .ydk file is valid or update the data and try again.")

        finally:
            # Clean up the temporary file
            if uploaded_file and file_path and os.path.exists(file_path):
                os.remove(file_path)
                os.rmdir(os.path.dirname(file_path))

    # Add helpful information
    st.sidebar.markdown("""
    ### About
    Visualizes Small World connections in your Yu-Gi-Oh! deck and recommends bridges.

    #### How to use:
    1. Upload the .ydk file of your deck
    2. View all visualizations
    3. Update card database if needed

    #### Visualizations:
    - **Bridges**: Shows the top Small World bridges for cards in your deck, ranked by bridge score
    - **Graph**: Visualizes card connections as a network
    - **Small World Connections**: Shows the Small World connections between cards
    - **Searchable Cards Heatmap**: Displays which cards are searchable from which other cards as a heat map.
                        The darker the color, the more bridges between those cards in the deck,
    """)

if __name__ == "__main__":
    main()
