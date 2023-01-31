# YGO-small-world

[Small World](https://www.db.yugioh-card.com/yugiohdb/card_search.action?ope=2&cid=16555&request_locale=en) is a *Yu-Gi-Oh!* card which is notorious for being difficult to understand. The idea is that you reveal a card from your hand, reveal a card from your deck with exactly one property in common with your original card out of attack, defense, level, type, and attribute, then reveal a third card also with exactly one property in common with the second card, and add that third card to your hand.

In theory, Small World can search any monster from your deck and add it to your hand. However, it may not be possible to bridge a card in your hand to the card that you want to search. The first card you reveal in your deck is referred to as the Small World *bridge* which connects the card you reveal in your hand and the card you add to your hand.

If you use Small World, it is generally desirable to include one or more dedicated Small World bridges that connects many cards in your deck so you have plenty of options for what to search starting from any card. However, such cards are difficult to find due to the many ways that cards can be considered connected via Small World. Because of the difficulty in optimizing a deck for Small World, there is a high barrier of entry to use the card.

The purpose of this repository is to assist in finding the best Small World bridges for any deck.

## Dataset

The card data `cardinfo.json` is obtained from the [Yu-Gi-Oh! API](https://ygoprodeck.com/api-guide/).

## Installation

*   Get the small world bridge finder source codes

```
git clone https://github.com/KennethJAllen/YGO-small-world
cd YGO-small-world
```

*   Install small world bridge finder dependencies

```
pip install -r requirements.txt
```

## Running the Code

*   The Small World bridge finder is run with the `small_world_bridge_finder.ipynb` notebook.

*   Run the cell under the `Import Functions` header first.

### Find Bridges From List of Names

*   Running the cell under the `Find Best Bridges From List of Names` markdwon will calculate the best bridges between cards from an example Mathmech deck.

*   To find the best bridges for your deck, replace the list `deck_monster_names` with the list of monsters names in your main deck. If there are any cards in your deck that are required to connect to the bridges, replace `required_target_names` with a list of those card's names. If not, you can replace it with the empy list `[]`.

*   Card names must be typed to exactly match the actual card name, including using capital letters and symbols. Exact card names can be found in the [Yu-Gi-Oh! card database](https://ygoprodeck.com/card-database/).

### Find Bridges From YDK File

A YDK file is a file containing card IDs from a deck. It is supported by programs such as YGOPRO, Duelingbook, and Dueling Nexus.

*   Running the cell under the `Find Best Bridges From YDK File` markdown will calculate the best bridges between cards from an example Mathmech deck from a YDK file.

*   To get a list of bridges from your YDK file, add your file to the `YGO-small-world` folder. Then replace `mathmech_deck.ydk` with the name of your YDK file. Note: Monsters in the side deck are counted as cards that are desierable to find connections to.

## Small World Mathematics
We can use [graph theory](https://en.wikipedia.org/wiki/Graph_theory) to calculate which cards can and cannot be searched starting from any card.

We can model the monsters in a deck as the vertices of an undirected graph $G$. Define an edge between monsters $i$ and $j$ if they are a valid connections between each other for Small World. e.g. [Ash Blossom](https://www.db.yugioh-card.com/yugiohdb/card_search.action?ope=2&cid=12950) and [Effect Veiler](https://www.db.yugioh-card.com/yugiohdb/card_search.action?ope=2&cid=8933) would have an edge connecting them because they the same attack but have different types, attributes, defense, and level. However, Ash Blossom and [Ghost Belle](https://www.db.yugioh-card.com/yugiohdb/card_search.action?ope=2&cid=13587) would not have an edge connecting them 

Let $N$ be the number of distinct monsters in your deck. Label the distinct monsters in your deck $1$ through $N$. e.g. 'Ash Blossom' $= 1$, 'Effect Veiler' $= 2$, 'Ghost Belle' $= 3$, etc. Let $M$ be the $N \times N$ adjacency matrix of the graph $G$. There is a one in entry $(i,j)$ in $M$ if there is an edge between monster $i$ and monster $j$, with zeros everywhere else. In other words, there is a $1$ in entry $(i,j)$ if monster $i$ and monster $j$ connect through Small World. In this case, entry $(1,2)$ would have a $1$ because Ash Blossom and Effect Veiler connect, but entry $(1,3)$ would be $0$ because Ash Blossom and Ghost Belle do not connect because they have the same level, attack, defense, and type.

**Theorem: Starting with monster $i$, you can search monster $j$ with Small World if and only if entry $(i,j)$ is non-zero in the matrix $M^2$.**

Proof: Entry $(i,j)$ in $M^2$ is equal to the number of paths of length $2$ from vertex $i$ to vertex $j$ in $G$ (see [properties of the adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix)). If entry $(i,j)$ is zero, then there are no bridges between monsters $i$ and $j$. If entry $(i,j)$ is non-zero, then there is at least one bridge from $i$ to $j$, so $j$ can be searched starting with $i$.

To create your own Small World adjacency matrix from a dataframe of monsters

### Example

Consider a Mathmech deck consisting of the monsters
```
'Ash Blossom & Joyous Spring',
'D.D. Crow',
'Effect Veiler',
'Ghost Belle & Haunted Mansion',
'Mathmech Addition',
'Mathmech Circular',
'Mathmech Diameter',
'Mathmech Multiplication',
'Mathmech Nabla',
'Mathmech Sigma',
'Mathmech Subtraction',
'Nibiru, the Primal Being',
'PSY-Frame Driver',
'PSY-Framegear Gamma'
```
Then the graph of connections via Small World can be visualized as follows.

![Mathmech Deck Graph](https://github.com/KennethJAllen/YGO-small-world/blob/main/images/mathmech-graph.png)

The adjacency matrix corresponding to cards in a Mathmech deck is the following matrix $M$.

![Mathmech Deck Adjacency Matrix](https://github.com/KennethJAllen/YGO-small-world/blob/main/images/mathmech-adjacency-matrix.jpg)

Squaring $M$, we get the matrix $M^2$.

![Mathmech Deck Adjacency Matrix Squared](https://github.com/KennethJAllen/YGO-small-world/blob/main/images/mathmech-adjacency-matrix-squared.jpg)

Every entry in the first column corresponding to [Mathmech Circular](https://www.db.yugioh-card.com/yugiohdb/card_search.action?ope=2&cid=17430) is non-zero except for the entry corresponding to Mathmech Multiplication, which means that Mathmech Circular can be searched with Small World starting from any monster in the deck except Mathmech Multiplication.

Moreover, the diagonal entries of $M^2$ are the number of connections a card has to another card in the deck. The diagonal entry corresponding to Effect Veiler is $7$, which means that Effect Veiler connects with $7$ cards in the deck.

### Generate Adjacency Matrix from YDK File

To generate an adjacency matrix from a YDK file, run the `ydk_to_df_adjacency_matrix(ydk_file, squared=False)` function. To generate the square of the adjacency matrix, set the optional parameter to `squared=True`. An example is given in the notebook `small_world_bridge_finder.ipynb`.
