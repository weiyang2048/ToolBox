"""Shared SVG chart primitives for the Markets page's tab
renderers — the one canonical home (user-approved
consolidation 2026-11-02).

    scale(v, d0, d1, r0, r1)   linear map from a data domain
                               to a pixel range
    line_path(xs, ys)          an SVG polyline path string
    svg_elems(w, h, body)      wrap a body in the responsive
                               <svg> frame both tabs use

us_xus and the_dollar each carried byte-identical local
copies of these three until the consolidation; new tab
renderers import them from here rather than growing another
copy.
"""


def scale(v, d0, d1, r0, r1):
    return r0 if d1 == d0 else r0 + (v - d0) / (d1 - d0) * (r1 - r0)


def line_path(xs, ys):
    return " ".join(
        ("M" if i == 0 else "L") + " %.1f %.1f" % (x, y)
        for i, (x, y) in enumerate(zip(xs, ys)))


def svg_elems(w, h, body):
    return ('<svg viewBox="0 0 %d %d" width="100%%" '
            'xmlns="http://www.w3.org/2000/svg" '
            'style="display:block;max-width:%dpx">%s</svg>'
            % (w, h, w, body))
