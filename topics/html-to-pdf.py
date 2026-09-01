import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

# Default topics directory: the directory containing this script
# (topics/html-to-pdf.py -> topics/)
DEFAULT_TOPICS_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description = "Render every reveal.js HTML slide deck under a topics "
                   "directory to a PDF alongside it."
)
parser.add_argument(
    "topics_dir",
    nargs = "?",
    default = DEFAULT_TOPICS_DIR,
    type = Path,
    help = f"Directory to search for HTML files (default: {DEFAULT_TOPICS_DIR})",
)
parser.add_argument(
    "--force",
    action = "store_true",
    help = "Re-render every PDF even if it's already up to date with its HTML",
)
args = parser.parse_args()

topics_dir = args.topics_dir.resolve()
html_files = sorted(topics_dir.glob("**/*.html"))

if not html_files:
    print(f"No HTML files found under {topics_dir}")

def is_stale(html_file: Path, pdf_file: Path) -> bool:
    if not pdf_file.exists():
        return True
    return html_file.stat().st_mtime > pdf_file.stat().st_mtime

pending = [
    (html_file, html_file.with_suffix(".pdf"))
    for html_file in html_files
    if args.force or is_stale(html_file, html_file.with_suffix(".pdf"))
]

skipped = len(html_files) - len(pending)
if skipped:
    print(f"Skipping {skipped} PDF(s) already up to date")

if pending:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        for html_file, pdf_file in pending:
            url = "file://" + str(html_file)

            print(f"Rendering {html_file} -> {pdf_file}")
            page.goto(url + "?print-pdf", wait_until = "networkidle")
            page.locator(".reveal.ready").wait_for()
            page.pdf(path = str(pdf_file), prefer_css_page_size = True)

        browser.close()
