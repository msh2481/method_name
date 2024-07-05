def parse_method(file_content, method_name="METHOD_NAME"):

    '''
    Parses the target function
    '''

    lines = file_content.split("\n")
    non_empty_lines = [line for line in lines if line.strip()]

    indented_lines = []
    for line in non_empty_lines:
        indent = len(line) - len(line.lstrip())
        indent_level = indent // 4 + (indent % 4 > 0)
        indented_lines.append((indent_level, line))

    method_start = None
    for i, (_, line) in enumerate(indented_lines):
        if method_name in line:
            method_start = i
            break

    if method_start is None:
        return None

    method_body = []
    method_indent = indented_lines[method_start][0]

    for indent, line in indented_lines[method_start:]:
        if indent > method_indent or line == indented_lines[method_start][1]:
            stripped_line = line[method_indent * 4 :]
            method_body.append(stripped_line)
        else:
            break

    return "\n".join(method_body)


if __name__ == "__main__":
    file_content = """
    class Kek:
        def some_function():
            pass

        def METHOD_NAME():
            print("This is the method body")
            for i in range(5):
                print(f"Line {i}")
            
            if True:
                print("Nested block")

    class Lol:
        pass

    def another_function():
        pass
    """

    result = parse_method(file_content, "METHOD_NAME")
    print(result)
