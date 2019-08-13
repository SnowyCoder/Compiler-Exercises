import networkx as nx
import fa


class DeterministicFiniteAutomaton(fa.FiniteAutomaton):
    def __init__(self, data=None):
        self.graph = nx.MultiDiGraph(data)

    def add_node(self, id, final: bool = False):
        self.graph.add_node(id, final=final)

    def add_nodes(self, ids):
        for x in ids:
            self.graph.add_node(x, final=False)

    def add_edge(self, from_node, to_node, char):
        if char is None:
            raise Exception("Cannot have null char link")

        for (fn, tn, data) in self.graph.out_edges(from_node, data=True):
            if data['char'] == char:
                raise Exception("node/char already used")

        self.graph.add_edge(from_node, to_node, char=char)

    def start(self, starting_node=0):
        return DFARunner(self.graph, starting_node)

    def run(self, text, starting_node=0):
        r = self.start(starting_node)

        for c in text:
            r.on_char(c)

        return r.state


class DFARunner:
    def __init__(self, graph: nx.DiGraph, start):
        self.graph = graph
        self.state = start

    def on_char(self, char):
        if self.state is None: return

        out_node = None

        for (from_edge, to_edge, data) in self.graph.out_edges(self.state, data=True):
            if data['char'] == char:
                out_node = to_edge
                break

        self.state = out_node

    @property
    def error(self):
        return self.state is None


if __name__ == '__main__':
    dfa = DeterministicFiniteAutomaton()

    # not|new
    #          -o-> 2 -t-> 3|
    # 0 -n-> 1 |
    #          -e-> 4 -w-> 5|

    dfa.add_nodes([0, 1, 2, 4])
    dfa.add_node(3, True)
    dfa.add_node(5, True)

    dfa.add_edge(0, 1, 'n')
    dfa.add_edge(1, 2, 'o')
    dfa.add_edge(2, 3, 't')

    dfa.add_edge(1, 4, 'e')
    dfa.add_edge(4, 5, 'w')

    print(dfa.run("not"))
    print(dfa.run("new"))
    print(dfa.run("now"))

