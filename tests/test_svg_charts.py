"""The shared SVG primitives: exact output contracts, since
two live tab renderers depend on them byte for byte."""
from toolbox.DataZoo.market.svg_charts import line_path, scale, svg_elems


def test_scale_linear_and_degenerate():
    assert scale(5, 0, 10, 0, 100) == 50
    assert scale(0, 0, 10, 200, 0) == 200
    assert scale(9, 3, 3, 7, 8) == 7  # degenerate domain -> r0


def test_line_path_format():
    assert line_path([0, 10], [1.04, 2.06]) == "M 0.0 1.0 L 10.0 2.1"
    assert line_path([], []) == ""


def test_svg_elems_frame():
    out = svg_elems(640, 200, "<rect/>")
    assert out.startswith('<svg viewBox="0 0 640 200" width="100%"')
    assert "max-width:640px" in out
    assert out.endswith("<rect/></svg>")
