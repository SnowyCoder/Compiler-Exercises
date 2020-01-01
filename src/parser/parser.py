import copy
from typing import List, Tuple, Optional
import networkx as nx

from scanner import nfa, dfa

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
        self.parent = parent  # type: SyntaxTreeNode
        self.ntype = ntype
        self.text = text
        self.children = []  # type: List[SyntaxTreeNode]


class SyntaxTree:
    def __init__(self):
        self.nodes = []  # type: List[SyntaxTreeNode]

    def add_node(self, node_type, parent: Optional[SyntaxTreeNode], text: str = None):
        node = SyntaxTreeNode(parent, node_type, text)
        self.nodes.append(node)

        if parent is not None:
            parent.children.append(node)

        return node

    @property
    def root(self) -> SyntaxTreeNode:
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


TOKEN_EOF = '__eof'


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

    def peek(self) -> Optional[Tuple[str, str]]:
        if self._fill_buffer():
            return self.buffer[-1]
        else:
            return TOKEN_EOF, ''

    def has_next(self) -> bool:
        return self._fill_buffer()

    def push_back(self, elem: Tuple[str, str]):
        self.buffer.append(elem)
        self.end_reached = False


TRANSFORM_LEFT_RECURSION = 0


class Grammar:
    def __init__(self, dictdef, root, result_transform=None):
        if result_transform is None:
            result_transform = []
        self.root = root
        self.dictdef = dictdef
        self.result_transofrm = result_transform

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

    def can_be_zero(self, focus):
        if focus[0] != NONTERMINAL:
            return False
        return self.next(focus[1], None) == []

    def resolve_all_left_recursion(self):
        # Only direct for now
        # TODO: create tree-rewrital after non-left-recursive grammar parsing (not implemented yet)
        dictdef = copy.deepcopy(self.dictdef)  # type: dict

        transformations = []

        for name, rules in self.dictdef.items():
            if any(rule[0][1] == name for rule in rules):
                # Left recursion!
                # First, discard A -> A
                def_list = dictdef[name]  # type: List

                pure_recursive_rule = [(NONTERMINAL, name)]

                recursive_rules = [rule[1:] for rule in def_list if rule[0] == (NONTERMINAL, name) if rule != pure_recursive_rule]
                non_recursive_rules = [rule for rule in def_list if rule[0] != (NONTERMINAL, name)]

                tail_name = name + '_TAIL'

                if tail_name in dictdef.keys():
                    # TODO: generate iterative name generator
                    raise Exception(f'Cannot generate non-left-recursive tree: {tail_name} already occupied')

                new_def_list = [x + [(NONTERMINAL, tail_name)] for x in non_recursive_rules]
                tail_def_list = [x + [(NONTERMINAL, tail_name)] for x in recursive_rules] + [[]]

                dictdef[name] = new_def_list
                dictdef[tail_name] = tail_def_list
                transformations.append((TRANSFORM_LEFT_RECURSION, name))

        return Grammar(dictdef, self.root, transformations)

    def apply_transformation(self, tree: SyntaxTree) -> SyntaxTree:
        res = SyntaxTree()

        left_transformed = frozenset([x[1] for x in self.result_transofrm if x[0] == TRANSFORM_LEFT_RECURSION])

        def run(input_node: SyntaxTreeNode, parent: Optional[SyntaxTreeNode]):
            if input_node.ntype[0] != NONTERMINAL or (input_node.ntype[1] not in left_transformed):
                output_node = res.add_node(input_node.ntype, parent, input_node.text)
                for c in input_node.children:
                    run(c, output_node)
                return

            # Let's rewrite
            node_type_name = input_node.ntype[1]
            tail_name = node_type_name + '_TAIL'

            # First: find the last nonempty tail
            current_tail = input_node
            while len(current_tail.children) > 0:
                current_tail = current_tail.children[-1]

            last_tail = current_tail.parent

            # Now we need to rotate this subtree using the last tail as the parent
            # And assigning the previous parent as the first child (do this recursively)

            def rotate_tree(current_input_node, output_parent):
                tail = current_input_node.ntype[1] == tail_name

                current_out_node = res.add_node((NONTERMINAL, node_type_name), output_parent)

                if tail:
                    # Tails should add their parent as their first child (as they are rotated)
                    rotate_tree(current_input_node.parent, current_out_node)

                # Ignore last element (the tree is rotated so now it's your father,
                # don't bother him or the belt shall come)
                for c in current_input_node.children[:-1]:
                    run(c, current_out_node)

            rotate_tree(last_tail, parent)

        run(tree.root, None)
        return res

    def parse(self, tokens: TokenStream) -> Optional[SyntaxTree]:
        tree = SyntaxTree()

        root = self.root
        focus = root
        focus_node = tree.add_node(focus, None)

        node_stack = [(None, None)]  # Last element

        token_iter = tokens

        current_token, current_text = None, None

        def advance_token(consume=True):
            nonlocal current_token, current_text
            if consume:
                token_iter.consume()
            current_token, current_text = token_iter.peek()

            if current_token == dfa.TOKEN_ERROR_NOT_RECOGNIZED:
                print(f'ERROR: Unrecognized token {current_text}')
                return advance_token()

            if current_token == TOKEN_EOF:
                while focus is not None:
                    if self.can_be_zero(focus):
                        consume_focus()
                    else:
                        raise Exception(f'Token stream terminates too early, expected {focus}')

        def consume_focus():
            nonlocal focus, focus_node
            focus, focus_node = node_stack.pop()

        if token_iter.peek()[0] == TOKEN_EOF:
            return None

        advance_token(False)

        while focus is not None:
            #print(f"STEP {focus} {current_token}")

            if current_token is 'space':
                advance_token()
                continue

            if focus[0] == TERMINAL:
                if current_token != focus[1]:
                    raise Exception(f'Error: expected {focus[1]} but found "{current_token}"')

                focus_node.text = current_text

                consume_focus()
                advance_token()

            else:
                children = self.next(focus[1], current_token)
                if children is None:
                    raise Exception(f'Unexpected token {current_token}')

                children_with_ids = [(x, tree.add_node(x, focus_node)) for x in children]

                node_stack += reversed(children_with_ids)
                consume_focus()
        return tree

    def __str__(self):
        res = ''
        for name, def_list in self.dictdef.items():
            name_p = f"{name} ->"
            res += name_p
            for i, d in enumerate(def_list):
                if i != 0:
                    res += len(name_p) * ' ' + '|'

                res += ' '
                res += ' '.join(x[1] for x in d)
                res += '\n'
        return res

    @classmethod
    def parse_grammar(cls, text):
        token_stream = TokenStream(iter(GRAMMAR_SCANNER.tokenize(text)))
        grammar = dict()
        names = set()
        first_name = None

        while True:
            tree = GRAMMAR_GRAMMAR.parse(token_stream)
            if tree is None:
                break

            if tree.root.children[0].ntype[1] == 'newline':
                # Empty expression
                continue

            name, definition = cls.parse_grammar_definition_from_tree(tree)

            if name in names:
                raise Exception(f'Double definition of {name}')

            if first_name is None:
                first_name = name

            names.add(name)
            grammar[name] = definition

        # Recognize terminals and nonterminals

        grammar2 = {
            name: [
                [(NONTERMINAL if comp_name in names else TERMINAL, comp_name) for comp_name in comp_names] for comp_names in def_list
            ] for name, def_list in grammar.items()
        }

        return Grammar(grammar2, (NONTERMINAL, first_name))

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
    input_grammar = """
    expr -> expr plus term
          | expr minus term
          | term 
    term -> term mul factor
          | term div factor
          | factor
    factor -> open_phar expr close_phar
            | number
    """

    grammar = Grammar.parse_grammar(input_grammar)

    input_scanner = nfa.Regex.regex_map_to_nfa({
        'space': "[ \t]+",
        'number': '[0123456789]+',
        'plus': '[+]',
        'minus': '-',
        'mul': '[*]',
        'div': '/',
        'open_phar': '[(]',
        'close_phar': ')',
    }).to_dfa().minimize()

    print(grammar)
    print('---- LEFT REC RESOLVED ----')
    grammar = grammar.resolve_all_left_recursion()
    print(grammar)

    token_stream = TokenStream(iter(input_scanner.tokenize('1+2*(3+4)')))
    gout = grammar.parse(token_stream)
    gout.visualize()
    gout2 = grammar.apply_transformation(gout)
    gout2.visualize()



