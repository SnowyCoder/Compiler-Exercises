from typing import List, Iterator, Tuple

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
        index = 0

        def consume_expected(chs):
            nonlocal index
            for ch in chs:
                if text[index] != ch:
                    raise Exception(f'Expected {ch} instead of {text[index]} at index {index}')
                index += 1

        def clear_spaces():
            nonlocal index
            while text[index] == ' ':
                index += 1

        def parse_identifier():
            nonlocal index

            clear_spaces()

            start_index = index
            while text[index] != ' ':
                index += 1
            return text[start_index:index]

        while text[index] != '\0':
            node_name = parse_identifier()
            # Arrow
            clear_spaces()
            consume_expected('->')
            clear_spaces()

            while text[index] != '|' and text[index] != '\0':
                neighbour = parse_identifier()


def parse(grammar: Grammar, tokens: Iterator[Tuple[str, str]]):
    tree = []

    def add_tree_node(node_type, parent, text=None) -> int:
        tree.append([parent, node_type, text])
        return len(tree) - 1

    root = grammar.root
    focus = root
    focus_index = add_tree_node(focus, 0)

    node_stack = [(None, -1)]  # Last element

    token_iter = tokens

    current_token, current_text = None, None

    def advance_token():
        nonlocal current_token, current_text
        current_token, current_text = next(token_iter, (None, None))

    def consume_focus():
        nonlocal focus, focus_index
        focus, focus_index = node_stack.pop()

    advance_token()

    while focus is not None:
        # print(f"STEP {focus} {current_token}")

        if current_token is 'space':
            advance_token()
            continue

        if focus[0] == TERMINAL:
            if current_token != focus[1]:
                raise Exception(f'Error: expected {focus[1]} but found "{current_token}"')

            tree[focus_index][2] = current_text
            advance_token()
            consume_focus()

            if current_token is None and focus is not None:
                raise Exception(f'Token stream terminates too early')
        else:
            children = grammar.next(focus[1], current_token)
            if children is None:
                raise Exception(f'Unexpected token {current_token}')

            children_with_ids = [(x, add_tree_node(x, focus_index)) for x in children]

            node_stack += reversed(children_with_ids)
            consume_focus()
    return tree


GRAMMAR_SCANNER = nfa.Regex.regex_map_to_nfa({
    'space': "[ \t\n]+",
    'identifier': '[abcdefghijklmnopqrstuvxyzABCDEFGHIJKLMNOPQRSTUVXYZ0123456789_-\']+',
    'arrow': '->',
    'separator': '[|]'
}).to_dfa().minimize()

# GRAMMAR_SCANNER.visualize()
# GRAMMAR_SCANNER.export('fuck')

GRAMMAR_GRAMMAR = Grammar({
    'EXPR': [
        [(TERMINAL, 'identifier'), (TERMINAL, 'arrow'), (NONTERMINAL, 'DEFINITION_LIST')]
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
        []
    ],
}, (NONTERMINAL, 'EXPR'))


def main():
    input_txt = """
    expr -> expr plus term
          | expr minus term
          | term 
    """

    token_stream = GRAMMAR_SCANNER.tokenize(input_txt)

    #print(list(token_stream))

    tree = parse(GRAMMAR_GRAMMAR, iter(token_stream))

    print('TREE')
    for index, v in enumerate(tree):
        print(index, v)


