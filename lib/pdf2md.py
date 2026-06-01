#!/usr/bin/env python3
"""
pdf2md.py — Convert PDF files to Markdown via pymupdf4llm (PyMuPDF).

Uses pymupdf4llm for high-quality text extraction with layout preservation,
table recognition, and optional image extraction.
"""

import sys
import os
from pathlib import Path

import pymupdf4llm


def convert(
    pdf_path: str,
    output_path: str | None = None,
    extract_images: bool = False,
) -> bool:
    """
    Convert a single PDF file to Markdown.

    Args:
        pdf_path:       Path to the .pdf file.
        output_path:    Path for the output .md (default: same name, .md extension).
        extract_images: If True, extract images into a sibling directory.

    Returns:
        True on success, False on failure.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"\033[31m  ✗ File not found:\033[0m {pdf_path}", file=sys.stderr)
        return False

    if output_path is None:
        output_path = str(pdf_path.with_suffix(".md"))

    output = Path(output_path)

    try:
        kwargs = {}

        if extract_images:
            # Create an images directory next to the output file
            img_dir = output.parent / f"{output.stem}_images"
            img_dir.mkdir(parents=True, exist_ok=True)
            kwargs["write_images"] = True
            kwargs["image_path"] = str(img_dir)

        md_text = pymupdf4llm.to_markdown(
            str(pdf_path),
            **kwargs,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(md_text, encoding="utf-8")
        return True

    except Exception as e:
        print(f"\033[31m  ✗ Conversion error:\033[0m {e}", file=sys.stderr)
        return False
