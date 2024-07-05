def parse_method(file_content, method_name="def METHOD_NAME"):
    """
    Parses the target function
    """

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
    arguments_ended = False

    for indent, line in indented_lines[method_start:]:
        if indent > method_indent or not arguments_ended:
            stripped_line = line[method_indent * 4 :]
            method_body.append(stripped_line)
            if ")" in line:
                arguments_ended = True
        else:
            break

    return "\n".join(method_body)


if __name__ == "__main__":
    file_content = """
def METHOD_NAME(
    img: np.ndarray,
    bgr2rgb=True,
    data_range=1.0,  # pylint: disable=unused-argument
    normalize=False,
    change_range=True,
    add_batch=True,
) -> np.ndarray:
    "Converts a numpy image array into a numpy Tensor array.
    Parameters:
        img (numpy array): the input image numpy array
        add_batch (bool): choose if new tensor needs batch dimension added
    "
    # check how many channels the image has, then condition. ie. RGB, RGBA, Gray
    # if bgr2rgb:
    #     img = img[
    #         :, :, [2, 1, 0]
    #     ]  # BGR to RGB -> in numpy, if using OpenCV, else not needed. Only if image has colors.
    if change_range:
""".strip()
    result = parse_method(file_content, "METHOD_NAME")
    print(result)
