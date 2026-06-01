#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
# MDPDF Installer
# Symlinks `pdf` and `md` commands to ~/.local/bin
# ──────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${HOME}/.local/bin"

# Colors
BOLD="\033[1m"
DIM="\033[2m"
RESET="\033[0m"
GREEN="\033[32m"
RED="\033[31m"
YELLOW="\033[33m"
CYAN="\033[36m"
MAGENTA="\033[35m"

echo ""
echo -e "${BOLD}${MAGENTA}  ╭─────────────────────────────────────╮${RESET}"
echo -e "${BOLD}${MAGENTA}  │   MDPDF Installer                   │${RESET}"
echo -e "${BOLD}${MAGENTA}  ╰─────────────────────────────────────╯${RESET}"
echo ""

# 1. Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo -e "  ${RED}✗ Error: python3 is not installed.${RESET}" >&2
    exit 1
fi

# 2. Check for PrinceXML (system dependency)
if ! command -v prince &> /dev/null; then
    echo -e "  ${YELLOW}⚠ Warning: 'prince' was not found in your PATH.${RESET}"
    echo -e "    PrinceXML is required for Markdown → PDF conversion ('pdf' command)."
    echo -e "    Download it from: ${CYAN}https://www.princexml.com/download/${RESET}"
    echo ""
else
    echo -e "  ${GREEN}✓${RESET} Found 'prince' (PrinceXML PDF Engine)"
fi

# 3. Check for Python dependencies
echo -e "  ${DIM}Checking Python packages...${RESET}"
MISSING_PACKAGES=()
for pkg in markdown latex2mathml pymdownx pymupdf4llm fitz; do
    if ! python3 -c "import $pkg" &> /dev/null; then
        MISSING_PACKAGES+=("$pkg")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "  ${YELLOW}⚠ Missing Python packages: ${MISSING_PACKAGES[*]}${RESET}"
    echo -e "    Installing dependencies via pip..."
    if command -v pip3 &> /dev/null; then
        pip3 install -r "$SCRIPT_DIR/requirements.txt"
        echo -e "  ${GREEN}✓${RESET} Dependencies installed successfully."
    elif command -v pip &> /dev/null; then
        pip install -r "$SCRIPT_DIR/requirements.txt"
        echo -e "  ${GREEN}✓${RESET} Dependencies installed successfully."
    else
        echo -e "  ${RED}✗ pip was not found.${RESET} Please run:"
        echo -e "    ${BOLD}pip3 install -r $SCRIPT_DIR/requirements.txt${RESET}"
    fi
else
    echo -e "  ${GREEN}✓${RESET} All Python dependencies are met."
fi
echo ""

# 4. Ensure bin dir exists
mkdir -p "$BIN_DIR"

# Make scripts executable
chmod +x "$SCRIPT_DIR/pdf"
chmod +x "$SCRIPT_DIR/md"

# Create symlinks (overwrite if exist)
ln -sf "$SCRIPT_DIR/pdf" "$BIN_DIR/pdf"
ln -sf "$SCRIPT_DIR/md"  "$BIN_DIR/md"

echo -e "  ${GREEN}✓${RESET} Linked: ${BOLD}pdf${RESET} → ${DIM}$SCRIPT_DIR/pdf${RESET}"
echo -e "  ${GREEN}✓${RESET} Linked: ${BOLD}md${RESET}  → ${DIM}$SCRIPT_DIR/md${RESET}"
echo ""

# Check if ~/.local/bin is in PATH
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
  echo -e "  ${YELLOW}⚠ $BIN_DIR is not in your PATH.${RESET}"
  echo "    Add this to your shell config (~/.bashrc or ~/.zshrc):"
  echo ""
  echo -e "      ${BOLD}export PATH=\"\$HOME/.local/bin:\$PATH\"${RESET}"
  echo ""
else
  echo -e "  ${GREEN}✓${RESET} $BIN_DIR is in PATH — you're all set!"
fi

echo -e "  ${BOLD}Usage:${RESET}"
echo -e "    ${CYAN}pdf file.md${RESET}              # Markdown → PDF"
echo -e "    ${CYAN}md  file.pdf${RESET}             # PDF → Markdown"
echo -e "    ${CYAN}pdf f1.md f2.md f3.md${RESET}    # Batch convert multiple .md"
echo -e "    ${CYAN}md  f1.pdf f2.pdf${RESET}        # Batch convert multiple .pdf"
echo ""
