from parser.parser import Grammar, TokenStream, SyntaxTree, SyntaxTreeNode
from scanner import nfa, dfa


def prepare_scanner() -> dfa.DeterministicFiniteAutomaton:
    return nfa.Regex.regex_map_to_nfa({
        'space': "[ \t]+",
        'number': '[0123456789]+',
        'plus': '[+]',
        'minus': '-',
        'mul': '[*]',
        'div': '/',
        'open_phar': '[(]',
        'close_phar': ')',
    }).to_dfa().minimize()


def prepare_grammar() -> Grammar:
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

    grammar = Grammar.parse_grammar(input_grammar)\
        .resolve_all_left_recursion()

    return grammar


def text_to_ast(scanner: dfa.DeterministicFiniteAutomaton, grammar: Grammar, text: str) -> SyntaxTree:
    token_stream = TokenStream(iter(scanner.tokenize(text)))
    gout = grammar.parse(token_stream)
    return grammar.apply_transformation(gout)


def interpret_ast(ast: SyntaxTree):
    def resolve_node(node: SyntaxTreeNode):
        t = node.ntype[1]
        if t == 'expr':
            if len(node.children) == 3:
                l = resolve_node(node.children[0])
                r = resolve_node(node.children[2])

                op = node.children[1].ntype[1]
                if op == 'plus':
                    return l + r
                elif op == 'minus':
                    return l - r
                else:
                    raise Exception(f"Unknown expr operation: {op}")
            else:
                return resolve_node(node.children[0])
        elif t == 'term':
            if len(node.children) == 3:
                l = resolve_node(node.children[0])
                r = resolve_node(node.children[2])

                op = node.children[1].ntype[1]
                if op == 'mul':
                    return l * r
                elif op == 'div':
                    return l / r
                else:
                    raise Exception(f"Unknown term operation: {op}")
            else:
                return resolve_node(node.children[0])
        elif t == 'factor':
            if len(node.children) == 3:
                return resolve_node(node.children[1])
            else:
                return resolve_node(node.children[0])
        elif t == 'number':
            return int(node.text)
        else:
            raise Exception(f"Unknown ntype: {node.ntype}")

    return resolve_node(ast.root)


def main():
    print("Compiling scanner...")
    scanner = prepare_scanner()
    print("Compiling grammar...")
    grammar = prepare_grammar()
    print("Ready!")

    while True:
        try:
            text = input("Expr: ")
        except EOFError:
            break

        ast = text_to_ast(scanner, grammar, text)
        res = interpret_ast(ast)
        print(res)



if __name__ == '__main__':
    main()
