"""Build the personal ML4ME study notebooks and saved browser reading versions.

Run with the ml4me-student interpreter from any directory. Lessons are authored
in the adjacent Markdown files. Every Python fence becomes an executable cell.
"""

from __future__ import annotations

import copy
import html
from pathlib import Path
import re
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


ROOT = Path(__file__).resolve().parent
LESSONS = [
    "01_foundations", "02_linear_models", "03_mathematical_tools",
    "04_probability_and_inference", "05_course_outlook",
]
PLOT_DESCRIPTIONS = {
    "01_foundations": ["Three measured extensions, the chosen straight-line prediction, and vertical residuals; the title reports mean squared error."],
    "02_linear_models": [
        "Training and validation observations with a fitted polynomial and the known synthetic relationship; the title compares both mean squared errors.",
        "Coefficient paths for squared-L2 and L1 penalties; L1 reaches exact zero for a coefficient while Ridge shrinks continuously.",
        "Squared, absolute, Huber and epsilon-insensitive regression losses as functions of the residual.",
        "Zero-one, perceptron, hinge, modified Huber and logistic classification losses as functions of the signed margin.",
        "A sigmoid maps a real-valued score to a probability between zero and one; a selected score is marked.",
    ],
    "03_mathematical_tools": [
        "A diagonal two-dimensional point cloud and its reconstruction from one principal-component coordinate.",
        "Gradient-descent weight values over successive updates, compared with the optimal weight of three.",
        "A small neural network approximates x squared; an adjacent plot shows training mean squared error over updates.",
    ],
    "04_probability_and_inference": [
        "Gaussian densities with three standard deviations; narrower distributions have taller peaks while preserving total area.",
        "A Beta prior and updated Beta posterior over an unknown success probability.",
        "Observed spring measurements, the posterior mean response, a latent-response uncertainty band and a wider predictive band.",
        "Target probabilities of four states compared with frequencies from a small Metropolis sampling experiment.",
    ],
    "05_course_outlook": [
        "Two illustrative samplers: one covers eight target clusters, while the other covers only two to illustrate mode collapse.",
        "A histogram of the affine transformation of Gaussian samples compared with its correctly normalized density.",
        "Three progressively noisier versions of a ring dataset illustrate the forward diffusion construction.",
    ],
}


def notebook_from_markdown(path):
    chunks = re.split(r"^```python\s*\n(.*?)^```\s*$", path.read_text(), flags=re.M | re.S)
    cells = []
    for index, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        factory = nbformat.v4.new_code_cell if index % 2 else nbformat.v4.new_markdown_cell
        cells.append(factory(chunk.strip()))
    return nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "ml4me-student", "display_name": "ML4ME (Python 3.11)", "language": "python"},
        "language_info": {"name": "python", "version": "3.11.8"},
    })


def widget_outputs(notebook, model_id, visited=None):
    visited = set() if visited is None else visited
    if model_id in visited:
        return []
    visited.add(model_id)
    models = notebook.metadata.get("widgets", {}).get("application/vnd.jupyter.widget-state+json", {}).get("state", {})
    state = models.get(model_id, {}).get("state", {})
    outputs = list(state.get("outputs", []))
    for child in state.get("children", []):
        outputs.extend(widget_outputs(notebook, child.replace("IPY_MODEL_", ""), visited))
    return outputs


def reading_copy(notebook):
    """Keep saved plots from Output widgets; leave live controls to Jupyter.

    This avoids depending on an external widget manager for reading the HTML.
    """
    result = copy.deepcopy(notebook)
    for cell in result.cells:
        if cell.cell_type != "code":
            continue
        outputs = []
        for output in cell.get("outputs", []):
            widget = output.get("data", {}).get("application/vnd.jupyter.widget-view+json")
            if widget:
                outputs.extend(nbformat.from_dict(item) for item in widget_outputs(notebook, widget["model_id"]))
            else:
                outputs.append(output)
        cell.outputs = outputs
    result.metadata.pop("widgets", None)
    return result


def verify_outputs(notebook):
    for cell in notebook.cells:
        if cell.cell_type == "code":
            for output in cell.outputs:
                if output.output_type == "error":
                    raise RuntimeError(f"{output.ename}: {output.evalue}")
    states = notebook.metadata.get("widgets", {}).get("application/vnd.jupyter.widget-state+json", {}).get("state", {})
    for model in states.values():
        for output in model.get("state", {}).get("outputs", []):
            if output.get("output_type") == "error":
                raise RuntimeError(f"Widget error: {output.get('ename')}: {output.get('evalue')}")


def write_html(notebook, name):
    exporter = HTMLExporter(template_name="lab")
    body, _ = exporter.from_notebook_node(reading_copy(notebook), resources={"metadata": {"name": name}})
    for description in PLOT_DESCRIPTIONS.get(name, []):
        body = body.replace('alt="No description has been provided for this image"',
                            'alt="' + html.escape(description, quote=True) + '"', 1)
    style = """<style>
    body.jp-Notebook { max-width: 1040px; margin: 0 auto; padding: 20px; box-sizing: border-box; }
    main { max-width: 900px; margin: 0 auto; }
    .jp-InputPrompt, .jp-OutputPrompt { display: none; }
    .jp-RenderedHTMLCommon { font-size: 16px; line-height: 1.75; }
    .jp-RenderedHTMLCommon h1 { font-size: 30px; line-height: 1.25; }
    .jp-RenderedHTMLCommon h2 { font-size: 23px; line-height: 1.35; margin-top: 2em; }
    .jp-RenderedHTMLCommon details { margin: 12px 0 24px; padding: 10px 16px; border-left: 3px solid #126782; background: #f3f7f8; }
    .jp-RenderedHTMLCommon summary { cursor: pointer; color: #126782; font-weight: 600; }
    .jp-RenderedHTMLCommon table { display: block; overflow-x: auto; font-size: 14px; }
    .jp-RenderedHTMLCommon th, .jp-RenderedHTMLCommon td { white-space: normal; }
    .jp-RenderedImage img { max-width: 100%; height: auto; }
    .study-nav { display: flex; flex-wrap: wrap; gap: 10px 20px; max-width: 900px; margin: 0 auto 24px; padding: 12px 0; border-bottom: 1px solid #d8e1e4; font: 15px/1.5 system-ui,sans-serif; }
    .study-nav a { color: #126782; text-decoration: none; }
    .study-nav a[aria-current="page"] { font-weight: 700; text-decoration: underline; }
    .study-footer { max-width: 900px; margin: 24px auto; font: 14px/1.6 system-ui,sans-serif; color: #52646e; }
    @media (max-width: 600px) { body.jp-Notebook { padding: 8px; } .jp-RenderedHTMLCommon { font-size: 16px; } .jp-RenderedHTMLCommon h1 { font-size: 26px; } }
    </style>"""
    pages = [("00_start_here", "Start here"), ("01_foundations", "1 · Foundations"),
             ("02_linear_models", "2 · Linear models"), ("03_mathematical_tools", "3 · Maths"),
             ("04_probability_and_inference", "4 · Probability"), ("05_course_outlook", "5 · Course outlook")]
    links = []
    for page, label in pages:
        current = ' aria-current="page"' if name == page else ""
        links.append(f'<a href="{page}.html"{current}>{html.escape(label)}</a>')
    nav = '<nav class="study-nav" aria-label="Study units">' + ''.join(links) + '</nav>'
    footer = '<footer class="study-footer">Personal ML4ME learning companion · Saved reading version. Open the matching notebook in Jupyter for live experiments.</footer>'
    body = body.replace("</head>", style + "</head>")
    body = re.sub(r"(<body[^>]*>)", lambda match: match[1] + nav, body, count=1)
    body = body.replace("</body>", nav + footer + "</body>")
    # Reading pages point to the published book; executable notebooks retain
    # their relative links to the original course files in this checkout.
    body = re.sub(r'href="\.\./((?:part1|part2|part3|notebooks|appendices|problems)/[^"#]+)\.(?:ipynb|qmd)"',
                  r'href="https://ideal.umd.edu/ML4ME_Textbook/\1.html"', body)
    body = body.replace('href="../START_HIER.md"',
                        'href="https://github.com/thewizard1248/ML4ME_Textbook/blob/main/START_HIER.md"')
    (ROOT / f"{name}.html").write_text(body)


def main():
    for name in LESSONS:
        start = time.monotonic()
        if "--render-only" in sys.argv:
            write_html(nbformat.read(ROOT / f"{name}.ipynb", as_version=4), name)
            continue
        notebook = notebook_from_markdown(ROOT / f"{name}.md")
        print(f"Executing {name}...", flush=True)
        NotebookClient(notebook, timeout=90, startup_timeout=60,
                       kernel_name="ml4me-student", resources={"metadata": {"path": str(ROOT)}}).execute()
        verify_outputs(notebook)
        nbformat.validate(notebook)
        nbformat.write(notebook, ROOT / f"{name}.ipynb")
        write_html(notebook, name)
        codes = sum(c.cell_type == "code" for c in notebook.cells)
        static = reading_copy(notebook)
        plots = sum("image/png" in o.get("data", {}) for c in static.cells if c.cell_type == "code" for o in c.outputs)
        print(f"PASS {name}: {codes} code cells, {plots} saved plots, {time.monotonic()-start:.1f}s", flush=True)
    write_html(notebook_from_markdown(ROOT / "00_start_here.md"), "00_start_here")


if __name__ == "__main__":
    main()
