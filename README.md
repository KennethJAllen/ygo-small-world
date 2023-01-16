# YGO-small-world

## Introduction
The purpose of this repository is to assist in finding the best Small World bridges for any particular deck in the card game *Yu-Gi-Oh*. 

[Small World](https://www.db.yugioh-card.com/yugiohdb/card_search.action?ope=2&cid=16555&request_locale=en) is a card which is notorious for being difficult to understand. The idea is that you reveal a card from your hand, reveal a card from your deck with exactly one property in common with your original card (out of attack, defense, level, type, and attribute), then reveal a third card also with exactly one property in common with the second card, and add that third card to your hand.

In theory, Small World can search any monster from your deck and add it to your hand. However, it may not be possible to bridge a card in your hand to the card that you want to search. The first card you reveal in your deck is referred to as the Small World *bridge* which connects the card you reveal in your hand and the card you add to your hand. It is desierable to include cards in your deck that act as bridges between many other cards, but finding an appropriate one can be difficult.

## Dataset

The card data `cardinfo.php.json` is obtained from the [Yu-Gi-Oh! API](https://ygoprodeck.com/api-guide/).

## Running the Code

The Small World bridge finder is run with the `sw-bridge-finder.ipynb` notebook.

The function `find_best_bridges` takes a list of monsters `deck_monster_names` in your main deck, as well as a list of monsters `required_target_names` that are required to connect to a particular bridge, either as search targets or starting cards. The outputs are the cards that connect the most cards in the deck in addition to connecting the required target cards.

Running the notebook will calculate the best bridges between cards from an example Mathmech deck that also connects the example targets. To find the best bridges for your deck, replace the list `deck_monster_names` with the list of monsters names in your main deck. If there are any cards in your deck that are required to connect to the output bridges, replace `required_target_names` with a list of those card's names. If not, you can replace it with the empy list `[]`.

Card names must be typed to exactly match the actual card name, including using capital letters and symbols. Exact card names can be found in the [Yu-Gi-Oh! card database](https://ygoprodeck.com/card-database/).

## Small World Mathematics
We can use [graph theory](https://en.wikipedia.org/wiki/Graph_theory) to calculate which cards can and cannot be searched starting from any card.

We can model the monsters in a deck as the vertices of an undirected graph $G$. Define an edge between monsters $i$ and $j$ if they are a valid connections between each other for Small World. e.g. [Ash Blossom](https://www.db.yugioh-card.com/yugiohdb/card_search.action?ope=2&cid=12950) and [Effect Veiler](https://www.db.yugioh-card.com/yugiohdb/card_search.action?ope=2&cid=8933) would have an edge connecting them because they the same attack but have different types, attributes, defense, and level. However, Ash Blossom and [Ghost Belle](https://www.db.yugioh-card.com/yugiohdb/card_search.action?ope=2&cid=13587) would not have an edge connecting them 

Let $N$ be the number of distinct monsters in your deck. Label the distinct monsters in your deck $1$ through $N$. e.g. Ash Blossom $= 1$, Effect Veiler $= 2$, Ghost Belle $= 3$, etc. Let $M$ be the $N \times N$ adjacency matrix of the graph $G$. There is a one in entry $(i,j)$ in $M$ if there is an edge between monster $i$ and monster $j$, with zeros everywhere else. In other words, there is a $1$ in entry $(i,j)$ if monster $i$ and monster $j$ connect through Small World. In this case, entry $(1,2)$ would have a $1$ because Ash Blossom and Effect Veiler connect, but entry $(1,3)$ would be $0$ because Ash Blossom and Ghost Belle do not connect because they have the same level, attack, and defense.

**Theorem: Starting with monster $i$, you can search monster $j$ with Small World if and only if entry $(i,j)$ is non-zero in the matrix $M^2$.**

Proof: Entry $(i,j)$ in $M^2$ is equal to the number of paths of length $2$ from vertex $i$ to vertex $j$ in $G$ (see [properties of the adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix)). If entry $(i,j)$ is zero, then there are no bridges between monsters $i$ and $j$. If entry $(i,j)$ is non-zero, then there is at least one bridge from $i$ to $j$, so $j$ can be searched starting with $i$.

### Example

Consider the adjacency matrix $M$ corresponding to cards in a Mathmech deck.

![Mathmech Deck Adjacency Matrix](https://github.com/KennethJAllen/YGO-small-world/blob/main/images/mathmech-adjacency-matrix.jpg)

Squaring $M$, we get the matrix $M^2$.

![Mathmech Deck Adjacency Matrix Squared](https://github.com/KennethJAllen/YGO-small-world/blob/main/images/mathmech-adjacency-matrix-squared.jpg)

Every entry in the first column corresponding to [Mathmech Circular](https://www.db.yugioh-card.com/yugiohdb/card_search.action?ope=2&cid=17430) is non-zero except for the entry corresponding to Mathmech Multiplication, which means that Mathmech Circular can be searched with Small World starting from any monster in the deck except Mathmech Multiplication.

Moreover, the diagonal entries of $M^2$ are the number of connections a card has to another card in the deck. The diagonal entry corresponding to Effect Veiler is $7$, which means that Effect Veiler connects with $7$ cards in the deck.
