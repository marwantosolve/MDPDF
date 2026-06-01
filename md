#!/usr/bin/env python3
"""
md — Convert PDF files to Markdown.

Usage:
  md file.pdf                        Convert file.pdf → file.md
  md file1.pdf file2.pdf file3.pdf   Batch convert multiple files
  md -o output.md file.pdf           Custom output name (single file only)
  md --images                        Also extract images
  md --help                          Show this help
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Add lib/ to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "lib"))

import pdf2md

# ──────────────────────────────────────────────────────────
# Colors
# ──────────────────────────────────────────────────────────
BOLD      = "\033[1m"
DIM       = "\033[2m"
RESET     = "\033[0m"
GREEN     = "\033[32m"
RED       = "\033[31m"
BLUE      = "\033[34m"
CYAN      = "\033[36m"
YELLOW    = "\033[33m"
MAGENTA   = "\033[35m"

def fmt_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def main():
    parser = argparse.ArgumentParser(
        prog="md",
        description="Convert PDF files to Markdown using pymupdf4llm.",
        epilog="Examples:\n"
               "  md paper.pdf\n"
               "  md ch1.pdf ch2.pdf ch3.pdf\n"
               "  md -o notes.md lecture.pdf\n"
               "  md --images textbook.pdf",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="One or more PDF (.pdf) files to convert.",
    )
    parser.add_argument(
        "-o", "--output",
        metavar="OUT",
        help="Output Markdown path (only valid with a single input file).",
    )
    parser.add_argument(
        "--images",
        action="store_true",
        help="Extract images alongside the Markdown output.",
    )

    args = parser.parse_args()

    # Validate
    if args.output and len(args.files) > 1:
        print(f"{RED}✗ --output can only be used with a single input file.{RESET}", file=sys.stderr)
        sys.exit(1)

    for f in args.files:
        if not f.lower().endswith(".pdf"):
            print(f"{YELLOW}⚠ Warning: '{f}' does not have a .pdf extension.{RESET}", file=sys.stderr)

    # Header
    n = len(args.files)
    print(f"\n{BOLD}{BLUE}  ╭─────────────────────────────────────╮{RESET}")
    print(f"{BOLD}{BLUE}  │{RESET}   {BOLD}PDF → MD{RESET}  {DIM}via PyMuPDF{RESET}            {BOLD}{BLUE}│{RESET}")
    print(f"{BOLD}{BLUE}  ╰─────────────────────────────────────╯{RESET}")
    print(f"  {DIM}Converting {n} file{'s' if n > 1 else ''}...{RESET}\n")

    success = 0
    failed = 0
    start_all = time.time()

    for f in args.files:
        name = Path(f).name
        out = args.output if len(args.files) == 1 else None
        t0 = time.time()

        print(f"  {CYAN}⟳{RESET} {name} {DIM}...{RESET}", end="", flush=True)

        ok = pdf2md.convert(f, output_path=out, extract_images=args.images)
        elapsed = time.time() - t0

        if ok:
            out_path = out or str(Path(f).with_suffix(".md"))
            size = fmt_size(Path(out_path).stat().st_size)
            print(f"\r  {GREEN}✓{RESET} {name} {DIM}→{RESET} {BOLD}{Path(out_path).name}{RESET}  {DIM}({size}, {elapsed:.1f}s){RESET}")
            success += 1
        else:
            print(f"\r  {RED}✗{RESET} {name}  {DIM}({elapsed:.1f}s){RESET}")
            failed += 1

    elapsed_all = time.time() - start_all

    # Summary
    print()
    if failed == 0:
        print(f"  {GREEN}{BOLD}✓ All {success} file{'s' if success > 1 else ''} converted successfully{RESET} {DIM}({elapsed_all:.1f}s){RESET}")
    else:
        print(f"  {YELLOW}{BOLD}⚠ {success} succeeded, {failed} failed{RESET} {DIM}({elapsed_all:.1f}s){RESET}")
    print()

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
