import time
from collections import defaultdict
from typing import Set

import networkx as nx
import fa
import nfa


class DeterministicFiniteAutomaton(fa.FiniteAutomaton):
    def __init__(self, data=None):
        self.graph = nx.MultiDiGraph(data)

    def add_node(self, id, token: str = None):
        self.graph.add_node(id, token=token)

    def add_nodes(self, ids):
        for x in ids:
            self.graph.add_node(x, token=None)

    def add_edge(self, from_node, to_node, char, double_ok=False):
        if char is None:
            raise Exception("Cannot have null char link")

        for (fn, tn, data) in self.graph.out_edges(from_node, data=True):
            if data['char'] == char:
                if not double_ok:
                    raise Exception("node/char already used")
                else:
                    return

        self.graph.add_edge(from_node, to_node, char=char)

    def minimize_hopcroft(self):
        node_to_partition = {}

        def partition_by_token(graph):
            out = defaultdict(list)
            for node, token in graph.nodes(data="token"):
                out[token].append(node)

            return list(frozenset(x) for x in out.values())

        def split(part: Set[int]):
            if len(part) == 1: return [part]
            splitted = defaultdict(list)

            for node in part:
                for _, out_node, char in self.graph.out_edges(node, data="char"):
                    if out_node in part:
                        # No need to split
                        splitted[0].append(node)
                    else:
                        out_part = node_to_partition[out_node]
                        splitted[(out_part, char)].append(node)

            return list(frozenset(x) for x in splitted.values())

        def update_ntp(partlist):
            for part in partlist:
                for node in part:
                    node_to_partition[node] = part
            return partlist

        todo_partition = update_ntp(partition_by_token(self.graph))
        partition = []

        while len(todo_partition) != len(partition):
            partition = todo_partition
            todo_partition = []

            for part in partition:
                splitted = update_ntp(split(part))
                todo_partition += splitted


        res = DeterministicFiniteAutomaton()

        next_id = 0
        for index, part in enumerate(partition):
            token = self.graph.nodes[next(iter(part))]["token"]
            res.add_node(next_id, token)
            next_id += 1

            for node in part:
                node_to_partition[node] = index

        for index, part in enumerate(partition):
            for node in part:
                for _, out_node, char in self.graph.out_edges(node, data="char"):
                    if out_node in part:
                        continue
                    res.add_edge(index, node_to_partition[out_node], char, double_ok=True)

        return res

    def reverse(self) -> 'nfa.NonDeterministicFiniteAutomaton':
        rev = nfa.NonDeterministicFiniteAutomaton()

        for node, token in self.graph.nodes(data='token'):
            rev.add_node(token)

        for nfrom, nto, nchar in self.graph.edges(data='char'):
            rev.add_edge(nto, nfrom, nchar)

        rev.begin, rev.end = rev.end, rev.begin

        return rev

    def minimize_brzozowski(self):
        return self.reverse().to_dfa().reverse().to_dfa()

    minimize = minimize_brzozowski

    def start(self, starting_node=0):
        return DFARunner(self.graph, starting_node)

    def run(self, text, starting_node=0):
        r = self.start(starting_node)

        for c in text:
            r.on_char(c)

        return r.token if not r.error else "ERROR"

    def tokenize(self, text):
        runner = self.start()
        token_start_index = 0
        last_successful_state = None
        last_successful_index = -1
        index = 0
        while index < len(text):
            runner.on_char(text[index])
            if runner.error:
                # Error, backtrack if you can
                if last_successful_state is not None:
                    # We can backtrack!
                    runner.state = last_successful_state
                    index = last_successful_index
                    yield (runner.token, text[token_start_index:index + 1])
                    token_start_index = index + 1
                    runner.reset()
                else:
                    # We can't backtrack, report error
                    yield (None, text[token_start_index: index + 1])
                    # What is the back-on-track strategy? well I don't really know, none for now
                    token_start_index = index + 1
                    runner.reset()
            else:
                token = runner.token
                if token is not None:
                    last_successful_state = runner.state
                    last_successful_index = index

            # Advance
            index += 1

        if runner.token is None:
            # No token recognized for final string
            yield (None, text[token_start_index:index])


class DFARunner:
    def __init__(self, graph: nx.DiGraph, start):
        self.graph = graph
        self.start = start
        self.state = start

    def on_char(self, char):
        if self.state is None: return None

        out_node = None

        for (from_edge, to_edge, data) in self.graph.out_edges(self.state, data=True):
            if data['char'] == char:
                out_node = to_edge
                break

        self.state = out_node
        if self.state is None: return None
        return self.token

    @property
    def error(self):
        return self.state is None

    @property
    def token(self):
        if self.error: raise Exception("Cannot get token on error")
        return self.graph.node[self.state]['token']

    def reset(self):
        self.state = self.start


if __name__ == '__main__':
    dfa = DeterministicFiniteAutomaton()

    # not|new
    #          -o-> 2 -t-> 3|
    # 0 -n-> 1 |
    #          -e-> 4 -w-> 5|

    dfa.add_nodes([0, 1, 2, 4])
    dfa.add_node(3, "NOT")
    dfa.add_node(5, "NEW")

    dfa.add_edge(0, 1, 'n')
    dfa.add_edge(1, 2, 'o')
    dfa.add_edge(2, 3, 't')

    dfa.add_edge(1, 4, 'e')
    dfa.add_edge(4, 5, 'w')

    print(dfa.run("not"))
    print(dfa.run("new"))
    print(dfa.run("now"))
    print(dfa.run("no"))

    print('---------- TEST 2 ----------')

    #    +-a->1-d-+
    # 0 -|        +->3
    #    +-b->2-d-+
    # Testing minimization algorithms
    dfa = DeterministicFiniteAutomaton()

    dfa.add_node(0, None)
    dfa.add_node(1, None)
    dfa.add_node(2, None)
    dfa.add_node(3, "E")

    dfa.add_edge(0, 1, 'a')
    dfa.add_edge(0, 2, 'b')
    dfa.add_edge(1, 3, 'd')
    dfa.add_edge(2, 3, 'd')

    dfa = dfa.minimize_hopcroft()
    print(dfa.run("ad"))
    print(dfa.run("bd"))
    print(dfa.run("aa"))
