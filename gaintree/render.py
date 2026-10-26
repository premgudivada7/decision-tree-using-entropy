"""Ways to display a fitted tree: plain text, Graphviz DOT, and a matplotlib figure.

All three accept `max_depth` to show only the top of a big tree; deeper parts
are replaced by a "..." node.
"""

from __future__ import annotations

import warnings
from typing import Optional

from .classifier import GainTreeClassifier, TreeNode


def _short(s, n: int = 30) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _counts_text(tree: GainTreeClassifier, node: TreeNode) -> str:
    if tree.n_classes_ <= 6:
        return ", ".join(f"{_short(c, 12)}:{int(k)}" for c, k in zip(tree.classes_, node.class_counts))
    return f"n={node.size}, {tree.n_classes_} classes"


def _verdict(tree: GainTreeClassifier, node: TreeNode) -> str:
    top = int(node.class_counts.max())
    return f"{_short(tree.classes_[node.majority])}  ({top}/{node.size})"


def _truncated(node: TreeNode, max_depth: Optional[int]) -> bool:
    return max_depth is not None and node.level >= max_depth and not node.is_terminal


def to_text(tree: GainTreeClassifier, details: bool = False,
            max_depth: Optional[int] = None) -> str:
    """Box-drawing view of the tree.

    [Outlook]
    ├── Overcast ⇒ Yes  (4/4)
    ├── Rain ⇒ [Wind]
    │   ├── Strong ⇒ No  (2/2)
    │   └── Weak ⇒ Yes  (3/3)
    └── Sunny ⇒ [Humidity]
        ...

    (x/n) = votes for the predicted class / rows in the leaf.
    details=True adds entropy H and information gain IG.
    """
    tree._check_fitted()

    def describe(node: TreeNode) -> str:
        if node.is_terminal:
            return _verdict(tree, node) + (f"  H={node.impurity:.3f}" if details else "")
        if _truncated(node, max_depth):
            return f"… {node.size} rows, mostly {_short(tree.classes_[node.majority])}"
        info = f"  H={node.impurity:.3f} IG={node.info_gain:.3f}" if details else ""
        return f"[{_short(tree.feature_names_[node.split_on])}]{info}"

    def label(edge: str) -> str:                    # "= Sunny" -> "Sunny", "<= 2.5" stays
        return _short(edge[2:] if edge.startswith("= ") else edge, 40)

    def children_of(node: TreeNode, pad: str):
        if node.is_terminal or _truncated(node, max_depth):
            return []
        edges = list(node.edges())
        out = []
        for k, (edge, child) in enumerate(edges):
            last = k == len(edges) - 1
            out.append((pad + ("└── " if last else "├── ") + label(edge) + " ⇒ " + describe(child),
                        child, pad + ("    " if last else "│   ")))
        return out

    lines = [describe(tree.trunk_)]
    stack = list(reversed(children_of(tree.trunk_, "")))   # iterative: no recursion limit
    while stack:
        text, node, pad = stack.pop()
        lines.append(text)
        stack.extend(reversed(children_of(node, pad)))
    return "\n".join(lines)


def to_dot(tree: GainTreeClassifier, max_depth: Optional[int] = None) -> str:
    """Graphviz DOT source (paste into https://dreampuf.github.io/GraphvizOnline)."""
    tree._check_fitted()
    out = ["digraph Tree {", 'node [shape=box, style="rounded,filled", fontname="Helvetica"];']
    ids = {}

    def nid(node):
        return ids.setdefault(id(node), len(ids))

    def esc(s):
        return str(s).replace("\\", "\\\\").replace('"', '\\"')

    stack = [tree.trunk_]
    while stack:
        node = stack.pop()
        i = nid(node)
        trunc = _truncated(node, max_depth)
        if node.is_terminal or trunc:
            head = _short(tree.classes_[node.majority]) + (" (...)" if trunc else "")
        else:
            head = _short(tree.feature_names_[node.split_on])
        label = (f"{esc(head)}\\nentropy = {node.impurity:.3f}\\nsamples = {node.size}"
                 f"\\n{esc(_counts_text(tree, node))}")
        color = "#d9f2d9" if node.is_terminal else ("#eeeeee" if trunc else "#e8eefc")
        out.append(f'{i} [label="{label}", fillcolor="{color}"];')
        if not node.is_terminal and not trunc:
            for edge, child in node.edges():
                out.append(f'{i} -> {nid(child)} [label="{esc(_short(edge, 40))}"];')
                stack.append(child)
    out.append("}")
    return "\n".join(out)


def draw(tree: GainTreeClassifier, ax=None, fontsize: int = 9,
              figsize: Optional[tuple] = None, max_depth: Optional[int] = None,
              max_leaves_shown: int = 64):
    """Draw the tree with matplotlib. Returns the Axes.

    Big trees are automatically cut to a depth that shows at most
    `max_leaves_shown` boxes on the bottom row (pass max_depth to choose).
    """
    import matplotlib.pyplot as plt

    tree._check_fitted()

    def visible_leaves(limit):
        return sum(1 for n in tree.walk()
                   if (n.is_terminal and (limit is None or n.level <= limit))
                   or (limit is not None and n.level == limit and not n.is_terminal))

    if max_depth is None and visible_leaves(None) > max_leaves_shown:
        max_depth = 0
        while visible_leaves(max_depth + 1) <= max_leaves_shown:
            max_depth += 1
        warnings.warn(f"Tree too big to draw fully; showing depth <= {max_depth}. "
                      "Pass max_depth=... to choose.", stacklevel=2)

    def kids(node):
        if node.is_terminal or _truncated(node, max_depth):
            return []
        return list(node.edges())

    # layout (iterative post-order): leaves get consecutive x slots,
    # parents sit above the middle of their children
    pos, next_x = {}, 0
    stack = [(tree.trunk_, False)]
    while stack:
        node, done = stack.pop()
        ch = kids(node)
        if not ch:
            pos[id(node)] = (next_x, -node.level)
            next_x += 1
        elif done:
            xs = [pos[id(c)][0] for _, c in ch]
            pos[id(node)] = ((min(xs) + max(xs)) / 2, -node.level)
        else:
            stack.append((node, True))
            stack.extend((c, False) for _, c in reversed(ch))

    n_slots = next_x
    shown_depth = max(-y for _, y in pos.values())
    if ax is None:
        figsize = figsize or (min(60, max(6, 2.2 * n_slots)), min(40, max(3, 1.8 * (shown_depth + 1))))
        _, ax = plt.subplots(figsize=figsize)

    palette = plt.get_cmap("tab10")
    for node in tree.walk():
        if id(node) not in pos:
            continue
        x, y = pos[id(node)]
        ch = kids(node)
        for label, child in ch:
            cx, cy = pos[id(child)]
            ax.plot([x, cx], [y, cy], color="#888", lw=1, zorder=1)
            ax.text((x + cx) / 2, (y + cy) / 2, _short(label, 20), fontsize=fontsize - 1,
                    ha="center", va="center", color="#444",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
        if ch:
            text = (f"{_short(tree.feature_names_[node.split_on], 20)}\nH={node.impurity:.3f}"
                    f"\nn={node.size}  IG={node.info_gain:.3f}")
            fc = "#eef2fb"
        else:
            counts = (str(node.class_counts.astype(int).tolist()) if tree.n_classes_ <= 4
                      else f"n={node.size}")
            more = "\n(...)" if not node.is_terminal else ""
            text = f"{_short(tree.classes_[node.majority], 20)}\nH={node.impurity:.3f}\n{counts}{more}"
            purity = node.class_counts.max() / node.class_counts.sum()
            a = 0.25 + 0.5 * purity          # blend with white -> opaque box
            fc = tuple(a * c + (1 - a) for c in palette(node.majority % 10)[:3])
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=2,
                bbox=dict(boxstyle="round,pad=0.4", fc=fc, ec="#555"))

    ax.set_xlim(-0.7, n_slots - 0.3)
    ax.set_ylim(-shown_depth - 0.6, 0.6)
    ax.axis("off")
    return ax


# Convenience methods on the estimator
def _attach():
    GainTreeClassifier.to_text = to_text
    GainTreeClassifier.to_dot = to_dot
    GainTreeClassifier.draw = draw
    GainTreeClassifier.__str__ = lambda self: (
        to_text(self) if hasattr(self, "trunk_") else repr(self))


_attach()

__all__ = ["to_text", "to_dot", "draw"]
