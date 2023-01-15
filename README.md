# YGO-small-world

## Introduction
[Small World](https://www.db.yugioh-card.com/yugiohdb/card_search.action?ope=2&cid=16555&request_locale=en) is a Yu-Gi-Oh card which is notorious for being difficult to understand. The idea is that you reveal a card from your hand, reveal a card from your deck with exactly one property in common with your original card (out of attack, defense, level, type, and attribute), then reveal a third card also with exactly one property in common with the second card, and add that third card to your hand.

In theory, Small World can search any monster from your deck. However, it may not be possible to bridge a card in your hand to the card that you want to search.

We can use [graph theory](https://en.wikipedia.org/wiki/Graph_theory) to calculate which cards can and cannot be searched starting from any card.

We can model the monsters in a deck as the vertices of an undirected graph $G$. Define an edge between monsters $i$ and $j$ if they are a valid connections between each other for Small World. e.g. [Ash Blossom](https://www.db.yugioh-card.com/yugiohdb/card_search.action?ope=2&cid=12950) and [Effect Veiler](https://www.db.yugioh-card.com/yugiohdb/card_search.action?ope=2&cid=8933) would have an edge connecting them because they the same attack but have different types, attributes, defense, and level. However, Ash Blossom and [Ghost Belle](https://www.db.yugioh-card.com/yugiohdb/card_search.action?ope=2&cid=13587) would not have an edge connecting them 

Let $N$ be the number of distinct monsters in your deck. Label the distinct monsters in your deck $1$ through $N$. e.g. Ash Blossom $= 1$, Effect Veiler $= 2$, Ghost Belle $= 3$, etc. Let $M$ be the $N \times N$ adjacency matrix of the graph $G$. There is a one in entry $(i,j)$ in $M$ if there is an edge between monster $i$ and monster $j$, with zeros everywhere else. In other words, there is a $1$ in entry $(i,j)$ if monster $i$ and monster $j$ connect through Small World. In this case, entry $(1,2)$ would have a $1$ because Ash Blossom and Effect Veiler connect, but entry $(1,3)$ would be $0$ because Ash Blossom and Ghost Belle do not connect because they have the same level, attack, and defense.

**Theorem: Starting with monster $i$, you can search monster $j$ with Small World if and only if entry $(i,j)$ is non-zero in the matrix $M^2$.**

Proof: Entry $(i,j)$ in $M^2$ is equal to the number of paths of length $2$ from vertex $i$ to vertex $j$ in $G$ (see [properties of the adjacency matrix](https://en.m.wikipedia.org/wiki/Adjacency_matrix)). If entry $(i,j)$ is zero, then there are no bridges between monsters $i$ and $j$. If entry $(i,j)$ is non-zero, then there is at least one bridge from $i$ to $j$, so $j$ can be searched starting with $i$.

### Example

Consider the adjacency matrix $M$ corresponding to cards in a Mathmech deck.

![Mathmech Deck Adjacency Matrix](https://i.imgur.com/8OwvRNZ.jpg)

Squaring $M$, we get the matrix $M^2$.

![Mathmech Deck Adjacency Matrix Squared](https://i.imgur.com/o7zqP7v.jpg)

Every entry in the first column corresponding to Mathmech Circular is non-zero except for the entry corresponding to Mathmech Multiplication, which means that Mathmech Circular can be searched with Small World starting from any monster in the deck except Mathmech Multiplication.

Moreover, the diagonal entries of $M^2$ are the number of connections a card has to another card in the deck. The diagonal entry corresponding to Effect Veiler is $7$, which means that Effect Veiler connects with $7$ cards in the deck.
