import networkx as nx


class FiniteAutomaton:
    def export(self, filename):
        g = self.graph.copy()

        for node in g.nodes:
            if g.nodes[node]['token'] is None:
                del g.nodes[node]['token']

        nx.write_graphml(g, filename + ".graphml")

    def visualize(self):
        import matplotlib.pyplot as plt
        from networkx.drawing.nx_agraph import graphviz_layout
        import pygraphviz

        pos = graphviz_layout(self.graph)
        plt.figure()
        nx.draw(self.graph, pos, edge_color="black", width=1, linewidths=1, node_size=500, node_color="pink", alpha=0.9,
                labels={
                    node: node if token is None else token for (node, token) in self.graph.nodes(data="token")
                })
        edge_labels = {(a, b): c for (a, b, c) in self.graph.edges(data="char")}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color="red")
        node_labels = {a: b for (a, b) in self.graph.nodes(data="char")}
        #nx.draw_networkx_labels(self.graph, pos, node_labels)

        plt.axis("off")
        plt.show()

    def alphabet(self):
        return set(char for (f, t, char) in self.graph.edges(data="char") if char is not "")
