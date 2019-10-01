from typing import List, Iterator, Tuple, Optional
import networkx as nx

from scanner import nfa

TERMINAL = 0
NONTERMINAL = 1


GRM = {
    'T': [
        [(NONTERMINAL, 'T'), (TERMINAL, '*'), (NONTERMINAL, 'F')],
        [(NONTERMINAL, 'T'), (TERMINAL, '/'), (NONTERMINAL, 'F')],
        [(NONTERMINAL, 'F')]
    ],
    'F': [
        [(TERMINAL, '('), (NONTERMINAL, 'T'), (TERMINAL, ')')],
        [(TERMINAL, 'NUM')]
    ]
}


class SyntaxTreeNode:
    def __init__(self, parent, ntype, text):
        self.parent = parent
        self.ntype = ntype
        self.text = text
        self.children = []


class SyntaxTree:
    def __init__(self):
        self.nodes = []  # type: List[SyntaxTreeNode]

    def add_node(self, node_type, parent: Optional[SyntaxTreeNode], text: str=None):
        node = SyntaxTreeNode(parent, node_type, text)
        self.nodes.append(node)

        if parent is not None:
            parent.children.append(node)

        return node

    @property
    def root(self):
        return self.nodes[0]

    def visualize(self):
        import matplotlib.pyplot as plt
        from networkx.drawing.nx_agraph import graphviz_layout
        import pygraphviz

        graph = nx.DiGraph()

        for ind, node in enumerate(self.nodes):
            graph.add_node(ind, text=node.ntype[1])

            if node.parent is not None:
                graph.add_edge(self.nodes.index(node.parent), ind)

            if node.text is not None:
                new_id = ind + len(self.nodes)
                graph.add_node(new_id, text=node.text)
                graph.add_edge(ind, new_id)

        pos = graphviz_layout(graph, prog='dot')
        plt.figure(figsize=(10, 10))
        nx.draw(graph, pos, edge_color="black", width=2, linewidths=1, node_size=200, node_color="pink", alpha=0.9,
                labels={
                    node: node if token is None else token for (node, token) in graph.nodes(data="text")
                })

        plt.axis("off")
        plt.show()


class TokenStream:
    def __init__(self, iterator):
        self.iterator = iterator
        self.buffer = []
        self.end_reached = False

    def assert_not_end(self):
        if self.end_reached:
            raise Exception('End Of Stream')

    def _fill_buffer(self):
        if len(self.buffer) > 0:
            return True

        if self.end_reached:
            return False

        next_element = next(self.iterator, None)
        if next_element is None:
            self.end_reached = True
            return False
        self.buffer.append(next_element)
        return True

    def consume(self) -> Tuple[str, str]:
        self._fill_buffer()
        self.assert_not_end()
        return self.buffer.pop()

    def peek(self, default=(None, None)) -> Optional[Tuple[str, str]]:
        if self._fill_buffer():
            return self.buffer[-1]
        else:
            return default

    def has_next(self) -> bool:
        return self._fill_buffer()

    def push_back(self, elem: Tuple[str, str]):
        self.buffer.append(elem)
        self.end_reached = False


class Grammar:
    def __init__(self, dictdef, root):
        self.root = root
        self.dictdef = dictdef

    def next(self, state, token):
        has_zero = False
        choice = None
        for arr in self.dictdef[state]:
            # print(arr)
            if len(arr) == 0:
                has_zero = True
            elif arr[0][0] == TERMINAL:
                if arr[0][1] == token:
                    if choice is not None:
                        raise Exception('Grammar is not LR(1)')
                    choice = arr
            else:
                ch = self.next(arr[0][1], token)
                if ch is not None and choice is not None:
                    raise Exception('Grammar is not LR(1)')
                choice = arr

        if choice is not None:
            return choice
        elif has_zero:
            return []
        else:
            return None

    @classmethod
    def parse_grammar(cls, text):
        token_stream = TokenStream(iter(GRAMMAR_SCANNER.tokenize(text)))
        grammar = dict()
        names = set()

        while True:
            tree = parse(GRAMMAR_GRAMMAR, token_stream)
            if tree is None:
                break

            if tree.root.children[0].ntype[1] == 'newline':
                # Empty expression
                continue

            name, definition = cls.parse_grammar_definition_from_tree(tree)

            if name in names:
                raise Exception(f'Double definition of {name}')

            names.add(name)
            grammar[name] = definition

        # Recognize terminals and nonterminals

        grammar2 = {
            name: [
                [(NONTERMINAL if comp_name in names else TERMINAL, comp_name) for comp_name in comp_names] for comp_names in def_list
            ] for name, def_list in grammar.items()
        }

        return grammar2

    @classmethod
    def parse_grammar_definition_from_tree(cls, tree: SyntaxTree):
        name_tree, arrow, def_list = tree.root.children

        name = name_tree.text

        def parse_name_list(node: SyntaxTreeNode):
            res = []
            while len(node.children) == 2:
                res.append(node.children[0].text)
                node = node.children[1]
            return res

        def parse_def_list(node: SyntaxTreeNode):
            res = [parse_name_list(node.children[0])]
            # Now only tails
            node = node.children[1]
            while node.children:
                # Tail has: separator, name_list, other tail
                res.append(parse_name_list(node.children[1]))
                node = node.children[2]
            return res

        return name, parse_def_list(def_list)


def parse(grammar: Grammar, tokens: TokenStream) -> Optional[SyntaxTree]:
    tree = SyntaxTree()

    root = grammar.root
    focus = root
    focus_node = tree.add_node(focus, None)

    node_stack = [(None, None)]  # Last element

    token_iter = tokens

    current_token, current_text = token_iter.peek()

    def advance_token():
        nonlocal current_token, current_text
        token_iter.consume()
        current_token, current_text = token_iter.peek()

    def consume_focus():
        nonlocal focus, focus_node
        focus, focus_node = node_stack.pop()

    if current_token is None:
        return None

    while focus is not None:
        # print(f"STEP {focus} {current_token}")

        if current_token is 'space':
            advance_token()
            continue

        if focus[0] == TERMINAL:
            if current_token != focus[1]:
                raise Exception(f'Error: expected {focus[1]} but found "{current_token}"')

            focus_node.text = current_text
            advance_token()
            consume_focus()

            if current_token is None and focus is not None:
                raise Exception(f'Token stream terminates too early')
        else:
            children = grammar.next(focus[1], current_token)
            if children is None:
                raise Exception(f'Unexpected token {current_token}')

            children_with_ids = [(x, tree.add_node(x, focus_node)) for x in children]

            node_stack += reversed(children_with_ids)
            consume_focus()
    return tree

GRAMMAR_SCANNER = nfa.Regex.regex_map_to_nfa({
    'space': "[ \t]+",
    'newline': "[\n]+",
    'identifier': '[abcdefghijklmnopqrstuvxyzABCDEFGHIJKLMNOPQRSTUVXYZ0123456789_-\']+',
    'arrow': '->',
    'separator': '[|]'
}).to_dfa().minimize()

# GRAMMAR_SCANNER.visualize()
# GRAMMAR_SCANNER.export('fuck')

GRAMMAR_GRAMMAR = Grammar({
    'EXPR': [
        [(TERMINAL, 'identifier'), (TERMINAL, 'arrow'), (NONTERMINAL, 'DEFINITION_LIST')],
        [(TERMINAL, 'newline')]
    ],
    'DEFINITION_LIST': [
        [(NONTERMINAL, 'NAME_LIST'), (NONTERMINAL, 'DEFINITION_LIST_TAIL')]
    ],
    'DEFINITION_LIST_TAIL': [
        [(TERMINAL, 'separator'), (NONTERMINAL, 'NAME_LIST'), (NONTERMINAL, 'DEFINITION_LIST_TAIL')],
        []
    ],
    'NAME_LIST': [
        [(TERMINAL, 'identifier'), (NONTERMINAL, 'NAME_LIST')],
        [(TERMINAL, 'newline')]
    ],
}, (NONTERMINAL, 'EXPR'))


def main():
    input_txt = """
    expr -> expr plus term
          | expr minus term
          | term 
    term -> term mul factor
          | term div factor
          | factor
    factor -> open_phar expr close_phar
            | number
    """

    print(Grammar.parse_grammar(input_txt))


