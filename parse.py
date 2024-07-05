def parse_method(file_content, method_name="METHOD_NAME"):
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

    for indent, line in indented_lines[method_start:]:
        if indent > method_indent or line == indented_lines[method_start][1]:
            stripped_line = line[method_indent * 4 :]
            method_body.append(stripped_line)
        else:
            break

    return "\n".join(method_body)


if __name__ == "__main__":
    file_content = """
#  python-holidays
#  ---------------
#  A fast, efficient Python library for generating country, province and state
#  specific sets of holidays on the fly. It aims to make determining whether a
#  specific date is a holiday as fast and flexible as possible.

from holidays.countries.dominican_republic import DominicanRepublic, DO, DOM
from tests.common import TestCase


class TestDominicanRepublic(TestCase):
    @classmethod
    def METHOD_NAME(cls):
        super().METHOD_NAME(DominicanRepublic)

    def test_country_aliases(self):
        self.assertCountryAliases(DominicanRepublic, DO, DOM)

    def test_2020(self):
        self.assertHolidays(
            ("2020-01-01", "Año Nuevo"),
            ("2020-01-06", "Día de los Santos Reyes"),
            ("2020-01-21", "Día de la Altagracia"),
            ("2020-01-26", "Día de Duarte"),
            ("2020-02-27", "Día de Independencia"),
            ("2020-04-10", "Viernes Santo"),
            ("2020-05-04", "Día del Trabajo"),
            ("2020-06-11", "Corpus Christi"),
            ("2020-08-16", "Día de la Restauración"),
            ("2020-09-24", "Día de las Mercedes"),
            ("2020-11-09", "Día de la Constitución"),
            ("2020-12-25", "Día de Navidad"),
        )
""".strip()
    result = parse_method(file_content, "METHOD_NAME")
    print(result)
