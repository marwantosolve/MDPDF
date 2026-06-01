#!/usr/bin/env python3
"""
md2pdf.py — Convert Markdown files to PDF via Prince.

Pipeline: .md → Python markdown (with extensions) → HTML (math as MathML) → Prince → .pdf

Math rendering uses latex2mathml to convert LaTeX to MathML server-side,
which Prince renders natively — no JavaScript needed.
"""

import sys
import os
import re
import subprocess
import tempfile
import markdown
import latex2mathml.converter
from pathlib import Path

# Directory where this script lives (for resolving style.css)
LIB_DIR = Path(__file__).resolve().parent

# ──────────────────────────────────────────────────────────
# LaTeX → MathML (server-side, for Prince)
# ──────────────────────────────────────────────────────────

def _preprocess_latex(latex: str) -> str:
    """Normalize LaTeX before sending to latex2mathml."""
    # \arg\max_X → \operatorname{argmax}_X
    latex = re.sub(r'\\arg\s*\\max\b', r'\\operatorname{argmax}', latex)
    # \arg\min_X → \operatorname{argmin}_X
    latex = re.sub(r'\\arg\s*\\min\b', r'\\operatorname{argmin}', latex)
    # \mathrel{+}= → += (latex2mathml chokes on \mathrel)
    latex = re.sub(r'\\mathrel\{([^}]*)\}', r'\1', latex)
    # \, (thin space) — ensure it doesn't break things
    latex = latex.replace(r'\,', r'\;')
    return latex


def _postprocess_mathml(mathml: str) -> str:
    """Clean up MathML output from latex2mathml."""
    # Remove stray backslashes leaked into <mi> or <mo> elements
    # e.g. <mi>\arg</mi> → <mi>arg</mi>
    mathml = re.sub(r'(<m[ion]>)\\([a-zA-Z]+)(</m[ion]>)', r'\1\2\3', mathml)
    return mathml


def _latex_to_mathml(latex: str, display: bool = False) -> str:
    """Convert a LaTeX expression to MathML markup."""
    try:
        latex = _preprocess_latex(latex)
        mathml = latex2mathml.converter.convert(latex)
        mathml = _postprocess_mathml(mathml)
        # Set display="block" for display math
        if display:
            mathml = mathml.replace('display="inline"', 'display="block"')
        return mathml
    except Exception:
        # Fallback: show raw LaTeX in a styled span
        delim = "$$" if display else "$"
        return f'<span class="math-fallback">{delim}{latex}{delim}</span>'


def _arithmatex_inline_formatter(math, **kwargs):
    """Custom pymdownx.arithmatex formatter for inline math → MathML."""
    return _latex_to_mathml(math, display=False)


def _arithmatex_block_formatter(math, **kwargs):
    """Custom pymdownx.arithmatex formatter for display math → MathML."""
    return f'<div class="math-display">{_latex_to_mathml(math, display=True)}</div>'


# ──────────────────────────────────────────────────────────
# Markdown → HTML conversion
# ──────────────────────────────────────────────────────────

EXTENSIONS = [
    "tables",
    "fenced_code",
    "codehilite",
    "toc",
    "meta",
    "footnotes",
    "attr_list",
    "def_list",
    "abbr",
    "md_in_html",
    "pymdownx.arithmatex",
    "pymdownx.superfences",
    "pymdownx.tasklist",
    "pymdownx.mark",
    "pymdownx.tilde",
    "pymdownx.caret",
    "pymdownx.smartsymbols",
]

EXTENSION_CONFIGS = {
    "pymdownx.arithmatex": {
        "generic": False,  # Use custom formatters instead of generic output
        "inline_syntax": ["dollar", "round"],
        "block_syntax": ["dollar", "square"],
    },
    "codehilite": {
        "css_class": "highlight",
        "guess_lang": False,
    },
    "pymdownx.tasklist": {
        "custom_checkbox": True,
    },
    "toc": {
        "permalink": False,
    },
}


def md_to_html(md_text: str) -> str:
    """Convert raw Markdown text to an HTML fragment with MathML."""
    # Pre-process: insert a blank line between consecutive display math blocks ($$ ... $$)
    # on adjacent lines. This prevents arithmatex from greedily combining them into a single block
    # and leaking raw dollar signs.
    md_text = re.sub(r"(\$\$[ \t]*\r?\n[ \t]*\$\$)", r"$$\n\n$$", md_text)

    md = markdown.Markdown(
        extensions=EXTENSIONS,
        extension_configs=EXTENSION_CONFIGS,
    )

    html = md.convert(md_text)

    # arithmatex (generic=False) produces this structure:
    #
    #   DISPLAY:
    #   <div class="arithmatex">
    #     <div class="MathJax_Preview">RAW LATEX</div>
    #     <script type="math/tex; mode=display">LATEX</script>
    #   </div>
    #
    #   INLINE:
    #   <span class="arithmatex">
    #     <span class="MathJax_Preview">RAW LATEX</span>
    #     <script type="math/tex">LATEX</script>
    #   </span>
    #
    # We must replace the ENTIRE wrapper (including the preview span that
    # leaks raw LaTeX) — not just the <script> tag.

    # 1) Display math: match the full <div class="arithmatex">...</div> wrapper
    def _replace_display(m):
        latex = m.group(1).strip()
        return f'<div class="math-display">{_latex_to_mathml(latex, display=True)}</div>'

    html = re.sub(
        r'<div\s+class="arithmatex">\s*'
        r'<div\s+class="MathJax_Preview">.*?</div>\s*'
        r'<script\s+type="math/tex;\s*mode=display">(.*?)</script>\s*'
        r'</div>',
        _replace_display,
        html,
        flags=re.DOTALL,
    )

    # 2) Inline math: match the full <span class="arithmatex">...</span> wrapper
    def _replace_inline(m):
        latex = m.group(1).strip()
        return _latex_to_mathml(latex, display=False)

    html = re.sub(
        r'<span\s+class="arithmatex">\s*'
        r'<span\s+class="MathJax_Preview">.*?</span>\s*'
        r'<script\s+type="math/tex">(.*?)</script>\s*'
        r'</span>',
        _replace_inline,
        html,
        flags=re.DOTALL,
    )

    # 3) Fallback: catch any leftover bare <script type="math/tex"> tags
    #    (in case arithmatex output format varies)
    def _replace_display_bare(m):
        latex = m.group(1).strip()
        return f'<div class="math-display">{_latex_to_mathml(latex, display=True)}</div>'

    html = re.sub(
        r'<script\s+type="math/tex;\s*mode=display">(.*?)</script>',
        _replace_display_bare,
        html,
        flags=re.DOTALL,
    )

    def _replace_inline_bare(m):
        latex = m.group(1).strip()
        return _latex_to_mathml(latex, display=False)

    html = re.sub(
        r'<script\s+type="math/tex">(.*?)</script>',
        _replace_inline_bare,
        html,
        flags=re.DOTALL,
    )

    # 4) Clean up any orphaned MathJax_Preview spans/divs that might remain
    html = re.sub(
        r'<(?:span|div)\s+class="MathJax_Preview">.*?</(?:span|div)>',
        '',
        html,
        flags=re.DOTALL,
    )

    # 5) Clean up residual $ / $$ delimiters left around MathML elements.
    #    Arithmatex sometimes wraps math correctly inside its spans but
    #    the original dollar-sign delimiters leak into the surrounding HTML
    #    as literal text.  Patterns:
    #      $<math ...>...</math>$    → strip both $
    #      $$<math ...>...</math>$$  → strip both $$
    #      $<math ...>...</math>$$   → strip both
    #      $$...$$  (bare, between display math blocks) → remove

    # a) Strip $/$$ immediately before <math
    html = re.sub(r'\${1,2}\s*(<math\b)', r'\1', html)
    # b) Strip $/$$ immediately after </math>
    html = re.sub(r'(</math>)\s*\${1,2}', r'\1', html)
    # c) Strip $/$$ immediately before <div class="math-display">
    html = re.sub(r'\${1,2}\s*(<div\s+class="math-display")', r'\1', html)
    # d) Strip $/$$ immediately after </div> for math-display
    html = re.sub(r'(</div>)\s*\${1,2}(?=\s*(?:<|$))', r'\1', html)
    # e) Remove bare $$ that are standalone (leftovers between display blocks)
    html = re.sub(r'(?:^|\n)\s*\$\$\s*(?:\n|$)', '\n', html)
    # f) Remove bare $$ inside <p> tags
    html = re.sub(r'<p>\s*\$\$\s*</p>', '', html)
    # g) Remove $$ at start/end of <p> content
    html = re.sub(r'(<p>)\s*\$\$\s*', r'\1', html)
    html = re.sub(r'\s*\$\$\s*(</p>)', r'\1', html)

    return html


def wrap_html(body: str, title: str = "Document", custom_css: str | None = None) -> str:
    """Wrap an HTML body fragment in a full HTML document with styles."""

    # Load default stylesheet
    default_css_path = LIB_DIR / "style.css"
    default_css = default_css_path.read_text() if default_css_path.exists() else ""

    # Load custom stylesheet if provided
    extra_css = ""
    if custom_css and Path(custom_css).exists():
        extra_css = Path(custom_css).read_text()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
{default_css}

/* MathML display math centering */
.math-display {{
  text-align: center;
  margin: 1em 0;
}}
.math-display math {{
  font-size: 1.15em;
}}
math[display="inline"] {{
  font-size: 1em;
}}
.math-fallback {{
  font-family: "JetBrains Mono", "Fira Code", monospace;
  color: #c7254e;
  background: #f8fafc;
  padding: 0.1em 0.3em;
  border-radius: 3px;
  font-size: 0.9em;
}}

{extra_css}
  </style>
</head>
<body>
  <article>
{body}
  </article>
</body>
</html>"""


# ──────────────────────────────────────────────────────────
# HTML → PDF via Prince
# ──────────────────────────────────────────────────────────

def html_to_pdf(html: str, output_path: str) -> bool:
    """Feed HTML to Prince and produce a PDF. Returns True on success."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(html)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["prince", tmp_path, "-o", output_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"\033[31m  ✗ Prince error:\033[0m {result.stderr.strip()}", file=sys.stderr)
            return False
        return True
    finally:
        os.unlink(tmp_path)


# ──────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────

def convert(md_path: str, output_path: str | None = None, custom_css: str | None = None) -> bool:
    """
    Convert a single Markdown file to PDF.

    Args:
        md_path:     Path to the .md file.
        output_path: Path for the output .pdf (default: same name, .pdf extension).
        custom_css:  Optional path to a custom CSS file.

    Returns:
        True on success, False on failure.
    """
    md_path = Path(md_path)
    if not md_path.exists():
        print(f"\033[31m  ✗ File not found:\033[0m {md_path}", file=sys.stderr)
        return False

    if output_path is None:
        output_path = str(md_path.with_suffix(".pdf"))

    md_text = md_path.read_text(encoding="utf-8")
    title = md_path.stem.replace("-", " ").replace("_", " ").title()

    html_body = md_to_html(md_text)
    full_html = wrap_html(html_body, title=title, custom_css=custom_css)

    return html_to_pdf(full_html, output_path)
