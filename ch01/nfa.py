from typing import Dict

import networkx as nx

from dfa import DeterministicFiniteAutomaton
import fa


class NonDeterministicFiniteAutomaton(fa.FiniteAutomaton):
    def __init__(self, data=None):
        self.graph = nx.MultiDiGraph(data)
        self.begin = None
        self.end = None
        self.nextid = 0

    def add_node(self, token: str = None):
        curr_id = self.nextid
        self.nextid += 1

        self.graph.add_node(curr_id, token=token)

        if self.begin is None:
            self.begin = curr_id
        self.end = curr_id

        return curr_id

    def add_nodes(self, ids):
        for x in ids:
            self.graph.add_node(x, token=None)

    def add_edge(self, from_node, to_node, char):
        self.graph.add_edge(from_node, to_node, char=("" if char is None else char))

    def add_nfa(self, nfa: 'NonDeterministicFiniteAutomaton'):
        our_start = self.nextid

        for (node, token) in nfa.graph.nodes(data="token", default=False):
            self.add_node(token)

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

        nodes = list(graph.nodes(data="token"))
        removed_nodes = []

        # Linear optimization pass
        for (index, (node, token)) in enumerate(nodes):
            if node in removed_nodes:
                continue

            out_edges = list(graph.out_edges(node, data="char"))

            if len(out_edges) == 1:
                (_node, next_node, next_char) = out_edges[0]
                if next_char == "" and len(graph.in_edges(next_node)) == 1:
                    # Found a 0-pass arc,
                    #  \       /
                    # - O --- O -
                    #  /       \
                    # If the connection between the nodes is empty (has no chars) we can optimize it by
                    # unifying the nodes
                    graph.remove_edge(node, next_node)
                    for (_node, next_next, next_char) in graph.out_edges(next_node, data="char"):
                        graph.add_edge(node, next_next, char=next_char)

                    if _node["token"] is not None:
                        if token is not None:
                            raise Exception('Multiple tokens with same text!')
                        node["token"] = _node["token"]

                    graph.remove_node(next_node)
                    removed_nodes.append(next_node)

        # Re-enumerate pass

        translation = {}

        for (index, (node, token)) in enumerate(nodes):
            if node not in removed_nodes:
                new_x = res.add_node(token=token)
                translation[index] = new_x

        for (f, t, c) in graph.edges(data="char"):
            res.add_edge(translation[f], translation[t], c)

        return res

    def to_dfa(self):
        res = DeterministicFiniteAutomaton()

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

        def get_token(nodes):
            tokens = set(filter(lambda x: x is not None, (self.graph.nodes[node]["token"] for node in nodes)))
            if len(tokens) == 0:
                return None
            elif len(tokens) == 1:
                return next(iter(tokens))
            else:
                raise Exception(f"Multiple tokens for the same text: '{','.join(tokens)}' ")

        alphabet = self.alphabet()

        start_nodes = nochar_closure(frozenset({self.begin}))
        out_nodes = {start_nodes: 0}
        work_list = [start_nodes]

        res.add_node(0, token=get_token(start_nodes))
        next_id = 1

        while work_list:
            current = work_list.pop()

            for char in alphabet:
                t = nochar_closure(advance(current, char))

                if not t: continue

                if t not in out_nodes:
                    out_nodes[t] = next_id
                    res.add_node(next_id, get_token(t))
                    next_id += 1
                    work_list.append(t)

                res.add_edge(out_nodes[current], out_nodes[t], char)
                # print(f"  {current} + {char} = {t}")
                #print(f"  {out_nodes[current]} + {char} = {out_nodes[t]}")
        return res

    @classmethod
    def from_text(cls, text: str, token: str):
        nfa = cls()
        last = nfa.add_node()

        for char in text:
            next = nfa.add_node()
            nfa.add_edge(last, next, char)
            last = next

        nfa.graph.nodes[last]['token'] = token

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
    def or_catenate_old(nfa1, nfa2):
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
    def or_catenate(*nfas):
        #       -> nfa1 |
        # first |       -> last
        #       -> nfa2 |
        res = NonDeterministicFiniteAutomaton()

        first = res.add_node()
        start = [res.add_nfa(nfa) for nfa in nfas]
        last = res.add_node()

        for x in range(len(nfas)):
            res.add_edge(first, start[x] + nfas[x].begin, None)
            res.add_edge(start[x] + nfas[x].end, last, None)

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
    def parse_group(text: str, token):
        nfa = None  # type: NonDeterministicFiniteAutomaton
        last_nfa = None
        next_token = 0

        def flush_last():
            nonlocal nfa, last_nfa

            if nfa is None:
                nfa = last_nfa
            elif last_nfa is not None:
                nfa = Regex.concatenate((nfa, last_nfa))

            last_nfa = None

        def allocate_token():
            nonlocal next_token
            tk = next_token
            next_token += 1
            return tk

        def transform_tokens():
            """
            The parsing method colors each generated nfa with a sequential token id, when the group parsing is done
            every token is removed expect from the last one that is replaced with the real token, this function does
            exactly that, iterating every node and trasforming its token
            """

            target_token = next_token - 1

            # If the last character is a star there isn't just one target token
            # But two because two nfas could be correct (the starred element and
            # the one before him)
            if len(text) > 0 and text[-1] == '*':
                target_token -= 1

            replaced_token = token
            for (node, ntoken) in nfa.graph.nodes(data="token"):
                if ntoken is None: continue
                if ntoken < target_token:
                    nfa.graph.nodes[node]["token"] = None
                else:
                    nfa.graph.nodes[node]["token"] = replaced_token

        # print(f"START GROUP '{text}'")

        i = 0
        while i < len(text):
            char = text[i]
            # print(char)

            if char == "(":
                flush_last()

                group_end = text.index(")", i + 1)
                subtext = text[i + 1: group_end]
                last_nfa = Regex.parse_group(subtext, allocate_token())
                i = group_end
            elif char == "*" or char == '+':
                # + and * only differ in the token assignment
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
                transform_tokens()
                next_nfa = Regex.parse_group(text[i + 1:], token)
                nfa = Regex.or_catenate(nfa, next_nfa)
                return nfa
            else:  # Text
                flush_last()
                last_nfa = NonDeterministicFiniteAutomaton.from_text(char, allocate_token())

            i += 1

        # print(f"DONE GROUP {text}")

        flush_last()
        transform_tokens()
        return nfa

    @staticmethod
    def regex_to_nfa(text: str, token):
        return Regex.parse_group(text, token)

    @staticmethod
    def regex_map_to_nfa(remap: Dict[str, str]):
        return Regex.or_catenate(*[Regex.regex_to_nfa(b, a) for (a, b) in remap.items()])


if __name__ == '__main__':
    nfa = Regex.regex_to_nfa("ad(b|c)*", "E")
    # nfa = nfa.simplify()
    # nfa.visualize()
    dfa = nfa.to_dfa()
    # dfa.visualize()  # Does not work well, self-edges missing and some edges end up overlapping
    print(dfa.run("adba"))
    print(dfa.run("adbcbcbc"))
    print(dfa.run("a"))
    print(dfa.run("ad"))
    print(dfa.run("adbcbcbca"))

    print('---------- TEST 2 ----------')

    nfa = Regex.regex_map_to_nfa({
        'SPACE': ' +',
        'NOT': 'not',
        'NOR': 'nor',
        'AND': 'and',
        'BINARY_NUMBER': 'b(0|1)+'
    })
    nfa.visualize()
    dfa = nfa.to_dfa()
    dfa.visualize()
    print(list(dfa.tokenize('not  b0 nor b1 and')))

