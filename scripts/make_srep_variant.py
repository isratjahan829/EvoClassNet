"""Produce the Scientific Reports house-style variant of the manuscript.

The numbered manuscript built by build_docx2.py follows the section order that was asked for:
Abstract, Introduction, Literature review, Materials and methods, Result analysis and discussion,
Future work, Conclusion. The journal's own template differs in two ways: headings carry no
numbers, and the Methods section sits after the Discussion rather than before the Results. Both
orders contain exactly the same text, so rather than maintaining two drafts this script rewrites
the generated file: it moves the Methods block (the heading and every paragraph, table and figure
under it) to just before the declarations, and strips the leading numbers from every heading.

Nothing is reworded, so cross-references such as "Section 3.2" would become stale; they are
rewritten to name the section instead.
"""
import copy
import re
import sys
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "EvoClassNet_manuscript.docx"
OUT = ROOT / "EvoClassNet_manuscript_SciRep_order.docx"

METHODS_H = "3. Materials and methods"
RESULTS_H = "4. Result analysis and discussion"
MOVE_BEFORE = "Data availability"

# Cross-references that name a number become references by name, because the numbers go away.
XREF = [
    (r"Sections 4\.1 to 4\.3", "the benchmark-wide analyses below"),
    (r"Sections 4\.5 to 4\.14", "the receptor-binding-domain case study"),
    (r"Section 4\.12", "the subsection on withholding more positions"),
    (r"Section 4\.15", "the subsection on coronavirus genomes"),
    (r"Section 4\.5", "the random-split results"),
    (r"Section 3\.2", "the structural data described in Methods"),
    (r"Sections 4\.1 to\s+4\.3", "the benchmark-wide analyses"),
]
HEAD_NUM = re.compile(r"^\d+(?:\.\d+)?\.?\s+")
DRAWING = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing"


def heading_index(kids, text, doc):
    """Index of the body child whose paragraph text is exactly `text`.

    The paragraph text is read through python-docx rather than lxml's itertext, which on a
    w:p yields each run's text more than once.
    """
    want = {id(par._element): par.text.strip() for par in doc.paragraphs}
    for i, el in enumerate(kids):
        if want.get(id(el)) == text:
            return i
    return None


def main():
    if not SRC.exists():
        sys.exit(f"{SRC} missing; run scripts/build_docx2.py first")
    doc = Document(str(SRC))
    body = doc.element.body
    kids = list(body)

    i_meth = heading_index(kids, METHODS_H, doc)
    i_res = heading_index(kids, RESULTS_H, doc)
    i_dest = heading_index(kids, MOVE_BEFORE, doc)
    if None in (i_meth, i_res, i_dest):
        sys.exit(f"could not locate sections: {i_meth=} {i_res=} {i_dest=}")
    if not i_meth < i_res < i_dest:
        sys.exit("unexpected section order in the source document")

    block = [copy.deepcopy(el) for el in kids[i_meth:i_res]]
    dest = kids[i_dest]
    for el in kids[i_meth:i_res]:
        body.remove(el)
    for el in block:
        dest.addprevious(el)

    # Assigning to run.text clears every child of the run, a w:drawing included, so a run is
    # only rewritten when the text actually changes and the run holds no drawing.
    def set_text(run, value):
        if value == run.text or run._element.findall(f".//{DRAWING}"):
            return
        run.text = value

    for para in doc.paragraphs:
        if not para.runs:
            continue
        if para.style.name.startswith("Heading"):
            set_text(para.runs[0], HEAD_NUM.sub("", para.runs[0].text))
        for run in para.runs:
            t = run.text
            for pat, name in XREF:
                t = re.sub(pat, name, t)
            set_text(run, t)

    doc.save(str(OUT))
    heads = [p.text for p in Document(str(OUT)).paragraphs
             if p.style.name == "Heading 1"]
    print("saved", OUT)
    print("order:", " | ".join(heads))


if __name__ == "__main__":
    main()
