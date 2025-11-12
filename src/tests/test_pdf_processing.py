import sys, os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../src/tests
SRC_DIR  = os.path.abspath(os.path.join(BASE_DIR, ".."))   # .../src
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

from component.pdf_processor import process_all_pdfs

pdf_folder = DATA / "FairFaxCounty" / "Item Sales Reports - Mar May 2025/Item Sales Reports - Mar May 2025"
output_file = DATA / "clean-data" / "sales.csv"

process_all_pdfs(pdf_folder, output_file)