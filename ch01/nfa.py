import networkx as nx

import dfa
import fa


class NonDeterministicFiniteAutomaton(fa.FiniteAutomaton):
    def __init__(self, data=None):
        self.graph = nx.MultiDiGraph(data)
        self.begin = None
        self.end = None
        self.nextid = 0

    def add_node(self, final: bool = False):
        curr_id = self.nextid
        self.nextid += 1

        self.graph.add_node(curr_id, final=final)

        if self.begin is None:
            self.begin = curr_id
        self.end = curr_id

        return curr_id

    def add_nodes(self, ids):
        for x in ids:
            self.graph.add_node(x, final=False)

    def add_edge(self, from_node, to_node, char):
        self.graph.add_edge(from_node, to_node, char=("" if char is None else char))

    def alphabet(self):
        return set(char for (f, t, char) in self.graph.edges(data="char") if char is not "")

    def add_nfa(self, nfa: 'NonDeterministicFiniteAutomaton'):
        our_start = self.nextid

        for (node, final) in nfa.graph.nodes(data="final", default=False):
            self.add_node(final)

        for (from_node, to_node, char) in nfa.graph.edges(data="char"):
            self.add_edge(our_start + from_node, our_start + to_node, char)

        return our_start

    def copy(self) -> "NonDeterministicFiniteAutomaton":
        res = NonDeterministicFiniteAutomaton()
        res.graph = self.graph.copy()
        res.begin = self.begin
        res.end = self.end
        return res

    def simplify(self):
        res = NonDeterministicFiniteAutomaton()

        graph = self.graph.copy()

        nodes = list(graph.nodes(data="final"))
        removed_nodes = []

        # Linear optimization pass
        for (index, (node, final)) in enumerate(nodes):
            if node in removed_nodes:
                continue

            out_edges = list(graph.out_edges(node, data="char"))

            if len(out_edges) == 1:
                (_node, next_node, next_char) = out_edges[0]
                if next_char == "" and len(graph.in_edges(next_node)) == 1:
                    graph.remove_edge(node, next_node)
                    for (_node, next_next, next_char) in graph.out_edges(next_node, data="char"):
                        graph.add_edge(node, next_next, char=next_char)
                    graph.remove_node(next_node)
                    removed_nodes.append(next_node)

                    out_edges = list(graph.out_edges(node, data="char"))

        # Re-enumerate pass

        translation = {}

        for (index, (node, final)) in enumerate(nodes):
            if node not in removed_nodes:
                new_x = res.add_node(final=final)
                translation[index] = new_x

        for (f, t, c) in graph.edges(data="char"):
            res.add_edge(translation[f], translation[t], c)

        return res

    def to_dfa(self):
        res = dfa.DeterministicFiniteAutomaton()

        def nochar_closure(nodes):
            """
            Follows the no-char paths from the nodes

            :param nodes: The initial nodes
            :return: The initial nodes with their no-char successors added
            """
            visited = set(nodes)
            work = list(nodes)

            while work:
                node = work.pop()

                for (_node, next_node, char) in self.graph.out_edges(node, data="char"):
                    if char != "" or next_node in visited: continue
                    visited.add(next_node)
                    work.append(next_node)

            return frozenset(visited)

        def advance(nodes, char):
            """
            Advances the state for every node using the character 'char', returns the new set of states

            Called 'Delta' in the book

            :param nodes: the initial states
            :param char: the character to advance with
            :return: the new states
            """
            out_nodes = set()
            for node in nodes:
                for (_node, next_node, jump_char) in self.graph.out_edges(node, data="char"):
                    if char != jump_char: continue
                    if next_node not in out_nodes:
                        out_nodes.add(next_node)

            return frozenset(out_nodes)

        alphabet = self.alphabet()

        start_nodes = nochar_closure(frozenset({self.begin}))
        out_nodes = {start_nodes: 0}
        work_list = [start_nodes]

        res.add_node(0)
        next_id = 1

        while work_list:
            current = work_list.pop()

            for char in alphabet:
                t = nochar_closure(advance(current, char))

                if not t: continue

                if t not in out_nodes:
                    out_nodes[t] = next_id
                    res.add_node(next_id)
                    next_id += 1
                    work_list.append(t)

                res.add_edge(out_nodes[current], out_nodes[t], char)
                # print(f"  {current} + {char} = {t}")
                #print(f"  {out_nodes[current]} + {char} = {out_nodes[t]}")
        return res



    @staticmethod
    def from_text(text: str):
        nfa = NonDeterministicFiniteAutomaton()
        last = nfa.add_node()

        for char in text:
            next = nfa.add_node()
            nfa.add_edge(last, next, char)
            last = next

        return nfa


class Regex:
    @staticmethod
    def concatenate(nfas):
        assert len(nfas) > 0
        res = NonDeterministicFiniteAutomaton()
        last_begin = res.add_nfa(nfas[0])

        for x in range(1, len(nfas)):
            nfa = nfas[x]
            last_nfa = nfas[x - 1]
            current_begin = res.add_nfa(nfa)

            res.add_edge(last_nfa.end + last_begin, nfa.begin + current_begin, None)

        return res

    @staticmethod
    def or_catenate(nfa1, nfa2):
        #       -> nfa1 |
        # first |       -> last
        #       -> nfa2 |
        res = NonDeterministicFiniteAutomaton()

        first = res.add_node()
        start1 = res.add_nfa(nfa1)
        start2 = res.add_nfa(nfa2)
        last = res.add_node()

        res.add_edge(first, start1 + nfa1.begin, None)
        res.add_edge(first, start2 + nfa2.begin, None)
        res.add_edge(start1 + nfa1.end, last, None)
        res.add_edge(start2 + nfa2.end, last, None)

        return res

    @staticmethod
    def star_repeat(nfa):
        res = NonDeterministicFiniteAutomaton()
        res.add_nfa(nfa)
        res.add_edge(res.end, res.begin, None)
        return res

    @staticmethod
    def for_repeat(nfa, times):
        return Regex.concatenate([nfa] * times)

    @staticmethod
    def parse_group(text: str):
        nfa = None  # type: NonDeterministicFiniteAutomaton
        last_nfa = None

        def flush_last():
            nonlocal nfa, last_nfa

            if nfa is None:
                nfa = last_nfa
            elif last_nfa is not None:
                nfa = Regex.concatenate((nfa, last_nfa))

            last_nfa = None

        # print(f"START GROUP '{text}'")

        i = 0
        while i < len(text):
            char = text[i]
            # print(char)

            if char == "(":
                flush_last()

                group_end = text.index(")", i + 1)
                subtext = text[i + 1: group_end]
                last_nfa = Regex.parse_group(subtext)
                i = group_end
            elif char == "*":
                assert last_nfa is not None
                last_nfa = Regex.star_repeat(last_nfa)
                flush_last()
            elif char == "{":
                assert last_nfa is not None
                number = ""
                while text[i] != "}":
                    number += text[i]
                    i += 1
                number = int(number)
                last_nfa = Regex.for_repeat(last_nfa, number)
            elif char == "|":
                flush_last()
                next_nfa = Regex.parse_group(text[i + 1:])
                nfa = Regex.or_catenate(nfa, next_nfa)
                break
            else:  # Text
                flush_last()
                last_nfa = NonDeterministicFiniteAutomaton.from_text(char)

            i += 1

        # print(f"DONE GROUP {text}")

        flush_last()
        return nfa

    @staticmethod
    def regex_to_nfa(text: str):
        return Regex.parse_group(text)


if __name__ == '__main__':
    nfa = Regex.regex_to_nfa("a(b|c)*")
    nfa = nfa.simplify()
    nfa.visualize()
    dfa = nfa.to_dfa()
    dfa.visualize()  # Does not work well, self-edges missing and some edges end up overlapping
    print(dfa.run("aba"))
    print(dfa.run("abcbcbc"))
    print(dfa.run("abcbcbca"))
