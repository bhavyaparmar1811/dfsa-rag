"""Show what Docling sees in a manifest document: one line per element.

A learning/debugging tool for the chunker. Each line is
`page | label | layer | text`, in Docling's reading order. Furniture (running
headers/footers) and text inside pictures are included but flagged, so you
can see exactly what the chunker will later drop.

    uv run python -m scripts.inspect_docling --doc dfsa-rulebook-cob --pages 30-32
    uv run python -m scripts.inspect_docling --doc dfsa-rulebook-cob --summary
    uv run python -m scripts.inspect_docling --doc dfsa-rulebook-cob --pages 32 --elements

`--pages` uses PDF page numbers (1-based, as a PDF viewer shows them), not the
page numbers printed in the rulebook's footer.
"""

import argparse
import logging
import sys
from collections import Counter

from docling_core.types.doc import ContentLayer, DoclingDocument, PictureItem, TableItem

from src.ingestion.elements import to_elements
from src.ingestion.manifest import load_manifest
from src.ingestion.parse import parse

ALL_LAYERS = set(ContentLayer)


def page_range(spec: str) -> range:
    start, _, end = spec.partition("-")
    return range(int(start), int(end or start) + 1)


def in_picture(item, doc: DoclingDocument) -> bool:
    parent = item.parent.resolve(doc) if item.parent else None
    while parent is not None:
        if isinstance(parent, PictureItem):
            return True
        parent = parent.parent.resolve(doc) if parent.parent else None
    return False


def describe(item, doc: DoclingDocument, width: int) -> str:
    if isinstance(item, TableItem):
        md = item.export_to_markdown(doc)
        return f"<table {len(item.data.table_cells)} cells> " + md[:width].replace("\n", " / ")
    if isinstance(item, PictureItem):
        caption = item.caption_text(doc)
        return f"<picture> {caption}" if caption else "<picture>"
    return getattr(item, "text", "").replace("\n", " ")[:width]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--doc", required=True, help="manifest id, e.g. dfsa-rulebook-cob")
    ap.add_argument("--pages", help="PDF page range, e.g. 30-32")
    ap.add_argument(
        "--elements",
        action="store_true",
        help="show the cleaned, re-ordered elements the chunker will see",
    )
    ap.add_argument("--summary", action="store_true", help="label counts for the whole document")
    ap.add_argument("--width", type=int, default=110, help="max characters of text per line")
    ap.add_argument("--force-reparse", action="store_true")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    entries = {e.id: e for e in load_manifest()}
    if args.doc not in entries:
        print(f"Unknown doc id {args.doc!r}. Known: {', '.join(entries)}")
        return 1
    doc = parse(entries[args.doc], force=args.force_reparse)

    items = [
        item
        for item, _level in doc.iterate_items(
            traverse_pictures=True, included_content_layers=ALL_LAYERS
        )
    ]

    if args.summary or not args.pages:
        counts = Counter(
            (item.label.value, item.content_layer.value, in_picture(item, doc)) for item in items
        )
        print(f"\n{args.doc}: {len(doc.pages)} pages, {len(items)} elements\n")
        print(f"{'label':<18}{'layer':<12}{'in picture':<12}count")
        for (label, layer, pic), n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"{label:<18}{layer:<12}{'yes' if pic else '':<12}{n}")
        if not args.pages:
            return 0

    pages = page_range(args.pages)

    if args.elements:
        print(f"\n{'page':<5}{'top':>6} {'left':>6}  {'label':<16}text")
        for el in to_elements(doc):
            if el.page in pages:
                text = el.text[: args.width].replace("\n", " / ")
                print(f"{el.page:<5}{el.top:>6.0f} {el.left:>6.0f}  {el.label:<16}{text}")
        return 0

    print(f"\n{'page':<5}{'label':<16}{'layer':<11}text")
    for item in items:
        page = item.prov[0].page_no if item.prov else None
        if page not in pages:
            continue
        flag = " [in picture]" if in_picture(item, doc) else ""
        print(
            f"{page:<5}{item.label.value:<16}{item.content_layer.value:<11}"
            f"{describe(item, doc, args.width)}{flag}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
