# 📄 MDPDF (md ⇄ pdf)

> **The ultimate command-line suite for high-fidelity, bidirectional document conversion.** 
> Seamlessly convert Markdown to stunning PDFs (with perfect LaTeX & MathML support via PrinceXML) and back to clean Markdown (using PyMuPDF).

<div align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/PRINCE-15-purple.svg" alt="PrinceXML Support">
</div>

---

## ✨ Features

- **⚡ Blazing Fast Bidirectional Conversion:**
  - **`pdf` command:** Convert `.md` ──> `.pdf` (uses PrinceXML).
  - **`md` command:** Convert `.pdf` ──> `.md` (uses PyMuPDF).
- **🧮 Perfect LaTeX Math Rendering:** Translates `$...$` and `$$...$$` math directly into native MathML server-side. PrinceXML renders it flawlessly with zero JavaScript required!
- **📐 Full Math Delimiter Support:** Handles advanced adjacent block display math, multi-line equations, and inline math with absolute precision.
- **🛠️ Smart LaTeX Normalization:** Automatically fixes and normalizes tricky LaTeX commands (e.g. `\arg\max` → `\operatorname{argmax}`, `\mathrel`, thin spaces) and repairs leaked backslashes before rendering.
- **📚 Batch Processing:** Pass multiple files to convert them all at once (e.g., `pdf ch1.md ch2.md ch3.md`).
- **🖼️ Image & Table Extraction:** Extract figures and inline tables cleanly when converting from PDF to Markdown.
- **🎨 Highly Customizable:** Inject your own custom CSS stylesheets to design premium, print-ready documents.

---

## 🚀 Installation

Getting started is quick and easy. Just clone this repository and run the installer:

```bash
# Clone the repository
git clone https://github.com/yourusername/MDPDF.git
cd MDPDF

# Run the installer
./install.sh
```

The installer will:
1. Validate your Python environment.
2. Check for the **PrinceXML** system engine.
3. Automatically install all required Python dependencies (`markdown`, `latex2mathml`, `pymupdf4llm`, etc.) via `pip`.
4. Create symbolic links in `~/.local/bin` so you can run `pdf` and `md` from anywhere!

### 📋 Prerequisites

- **Python 3.8+**
- **PrinceXML** (Required for the `pdf` command to compile HTML to PDF).
  - *Debian/Ubuntu:* `sudo apt install prince` (or download the latest package from [PrinceXML](https://www.princexml.com/download/))
  - *Fedora/RHEL:* `sudo dnf install prince`
  - *macOS:* `brew install prince`

---

## 💡 Usage

### 1. Markdown to PDF (`pdf`)

Convert a single Markdown file to a styled PDF:
```bash
pdf notes.md
```
*Creates `notes.pdf` in the same directory using the default print-optimized theme.*

#### 📁 Batch Conversion
Convert multiple files at once:
```bash
pdf chapter1.md chapter2.md chapter3.md
```

#### 🎨 Custom Stylesheet
Apply a custom CSS file to style your PDF:
```bash
pdf --style academic.css thesis.md
```

#### 📍 Custom Output Name
```bash
pdf -o final_report.pdf README.md
```

---

### 2. PDF to Markdown (`md`)

Convert a PDF file back into clean Markdown with preserved layouts:
```bash
md research_paper.pdf
```
*Creates `research_paper.md` in the same directory.*

#### 🖼️ Extract Images
Extract all embedded images into a dedicated folder next to the generated Markdown file:
```bash
md --images textbook.pdf
```
*Generates `textbook.md` and saves all images inside `textbook_images/`.*

#### 📁 Batch Conversion
```bash
md slide1.pdf slide2.pdf slide3.pdf
```

---

## 🛠️ How It Works (Under the Hood)

### MD ──> PDF Pipeline:
1. **Parser:** Python Markdown processes your text and extracts LaTeX formulas using `pymdown-extensions`.
2. **Converter:** Formulas are pre-processed and converted to native **MathML** via `latex2mathml` server-side.
3. **Styler:** Wraps the resulting HTML with a typography-optimized default stylesheet (`lib/style.css`) or your custom CSS.
4. **Compiler:** Passes the final clean HTML + MathML bundle to **PrinceXML**, which builds a pixel-perfect, print-quality PDF.

### PDF ──> MD Pipeline:
1. Uses `pymupdf4llm` to read the PDF stream.
2. Recognizes structural elements like headings, bulleted lists, standard paragraphs, and tables.
3. Extracts images and embeds them with markdown links.
4. Writes out a clean, readable `.md` file.

---

## 📦 Repository Structure

```
MDPDF/
├── lib/
│   ├── md2pdf.py      # Core Markdown-to-PDF conversion logic
│   ├── pdf2md.py      # Core PDF-to-Markdown conversion logic
│   └── style.css      # Default print-ready stylesheet
├── pdf                # CLI entrypoint for 'pdf' command
├── md                 # CLI entrypoint for 'md' command
├── install.sh         # Installer and system linker
├── requirements.txt   # Python dependencies
└── .gitignore         # Version control exclusion file
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
