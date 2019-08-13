import networkx as nx


class FiniteAutomaton:
    def export(self, filename):
        nx.write_graphml(self.graph, filename + ".graphml")

    def visualize(self):
        import matplotlib.pyplot as plt
        from networkx.drawing.nx_agraph import graphviz_layout
        import pygraphviz

        pos = graphviz_layout(self.graph)
        plt.figure()
        nx.draw(self.graph, pos, edge_color="black", width=1, linewidths=1, node_size=500, node_color="pink", alpha=0.9,
                labels={
                    node: node for node in self.graph.nodes()
                })
        edge_labels = {(a, b): c for (a, b, c) in self.graph.edges(data="char")}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color="red")

        plt.axis("off")
        plt.show()