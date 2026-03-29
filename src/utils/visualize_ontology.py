import json
from html import escape


def _count_subtree_docs(node):
    total = len(node.get("doc_ids", []))
    for child in node.get("children", []) or []:
        total += _count_subtree_docs(child)
    return total


def _render_node(node):
    name = escape(str(node.get("name", "Unnamed")))
    desc = escape(str(node.get("desc", "")).strip())
    level = int(node.get("level", 0))
    direct_docs = len(node.get("doc_ids", []) or [])
    subtree_docs = _count_subtree_docs(node)
    children = node.get("children", []) or []

    meta = (
        f'<div class="meta">level {level} | direct docs {direct_docs} | '
        f"subtree docs {subtree_docs} | children {len(children)}</div>"
    )
    desc_html = f'<div class="desc">{desc}</div>' if desc else ""
    children_html = ""
    if children:
        rendered_children = "\n".join(_render_node(child) for child in children)
        children_html = f'<ul class="children">\n{rendered_children}\n</ul>'

    return (
        "<li>"
        f'<div class="node-card"><div class="title">{name}</div>{meta}{desc_html}</div>'
        f"{children_html}"
        "</li>"
    )


def generate_html(json_path: str, html_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        root = json.load(f)

    total_nodes = 0
    max_depth = 0

    def walk(node):
        nonlocal total_nodes, max_depth
        total_nodes += 1
        max_depth = max(max_depth, int(node.get("level", 0)))
        for child in node.get("children", []) or []:
            walk(child)

    walk(root)
    total_docs = _count_subtree_docs(root)
    body = _render_node(root)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ontology Visualization</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #d8cfc0;
      --accent: #b45309;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      padding: 32px;
      background:
        radial-gradient(circle at top left, rgba(180, 83, 9, 0.12), transparent 28%),
        linear-gradient(180deg, #f7f3ea 0%, #efe7d9 100%);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }}
    .page {{
      max-width: 1180px;
      margin: 0 auto;
    }}
    .hero {{
      background: rgba(255, 253, 248, 0.88);
      border: 1px solid rgba(216, 207, 192, 0.9);
      border-radius: 20px;
      padding: 24px 28px;
      box-shadow: 0 18px 40px rgba(64, 39, 17, 0.08);
      backdrop-filter: blur(4px);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      line-height: 1.1;
    }}
    .summary {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 14px;
    }}
    .pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.8);
      font-size: 14px;
    }}
    .tree {{
      margin-top: 24px;
      padding: 0;
    }}
    .tree ul {{
      list-style: none;
      margin: 14px 0 0 22px;
      padding-left: 22px;
      border-left: 2px solid rgba(216, 207, 192, 0.9);
    }}
    .tree li {{
      list-style: none;
      margin: 0;
      padding: 10px 0 0;
      position: relative;
    }}
    .tree li::before {{
      content: "";
      position: absolute;
      left: -22px;
      top: 28px;
      width: 18px;
      border-top: 2px solid rgba(216, 207, 192, 0.9);
    }}
    .node-card {{
      background: var(--panel);
      border: 1px solid rgba(216, 207, 192, 0.95);
      border-radius: 16px;
      padding: 14px 16px;
      box-shadow: 0 10px 24px rgba(64, 39, 17, 0.06);
    }}
    .title {{
      font-size: 19px;
      font-weight: 700;
      color: #111827;
    }}
    .meta {{
      margin-top: 6px;
      color: var(--accent);
      font-size: 13px;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }}
    .desc {{
      margin-top: 9px;
      color: var(--muted);
      line-height: 1.55;
      font-size: 15px;
    }}
    @media (max-width: 768px) {{
      body {{
        padding: 16px;
      }}
      .hero {{
        padding: 18px;
      }}
      h1 {{
        font-size: 28px;
      }}
      .tree ul {{
        margin-left: 14px;
        padding-left: 14px;
      }}
      .tree li::before {{
        left: -14px;
        width: 10px;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Ontology Visualization</h1>
      <div class="summary">
        <div class="pill">nodes {total_nodes}</div>
        <div class="pill">max depth {max_depth}</div>
        <div class="pill">subtree attachments {total_docs}</div>
      </div>
    </section>
    <section class="tree">
      <ul>
        {body}
      </ul>
    </section>
  </div>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
