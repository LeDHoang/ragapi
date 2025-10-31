#!/usr/bin/env python3
import io
import re
import sys
from pathlib import Path
from typing import List, Tuple
from pypdf import PdfReader, PdfWriter

from Convert_excel_pdf.service.lool_client import LoolClient


def _sort_key_by_prefix(fname: str) -> Tuple[int, str]:
    # Prefer numeric prefix like "001_SheetName.xlsx"
    m = re.match(r"^(\d+)[_\-\s].*$", fname)
    if m:
        try:
            return (int(m.group(1)), fname.lower())
        except ValueError:
            pass
    return (10**9, fname.lower())


def find_sheet_excels(sheet_dir: Path) -> List[Path]:
    # Collect .xlsx files in sheet_dir and sort by numeric prefix if present
    files = [p for p in sheet_dir.iterdir() if p.is_file() and p.suffix.lower() == ".xlsx"]
    files.sort(key=lambda p: _sort_key_by_prefix(p.name))
    return files


def convert_sheets_and_combine(
    sheet_excels: List[Path],
    pdf_dir: Path,
    combined_pdf: Path,
    lool_url: str = "http://localhost:9980",
    endpoint: str = "/lool/convert-to/pdf",
    timeout: int = 1200
) -> Path:
    if not sheet_excels:
        raise ValueError("No sheet .xlsx files found")

    pdf_dir.mkdir(parents=True, exist_ok=True)
    client = LoolClient(base_url=lool_url, endpoint_path=endpoint, timeout=timeout)
    merger = PdfWriter()

    for idx, xlsx_path in enumerate(sheet_excels, 1):
        print(f"[{idx}/{len(sheet_excels)}] Converting sheet file: {xlsx_path.name}")
        pdf_bytes = client.convert_all_sheets(xlsx_path)

        # Read returned PDF (should be 1 page if the sheet file has only one sheet)
        reader = PdfReader(io.BytesIO(pdf_bytes))
        if len(reader.pages) == 0:
            print(f"  → Skipping empty PDF for {xlsx_path.name}")
            continue

        # Individual PDF path: use the sheet file's base name with .pdf
        pdf_filename = f"{xlsx_path.stem}.pdf"
        pdf_path = pdf_dir / pdf_filename

        # Save single-page (or first page if multiple) to disk
        single_writer = PdfWriter()
        single_writer.add_page(reader.pages[0])
        with open(pdf_path, "wb") as f:
            single_writer.write(f)
        print(f"  → Saved {pdf_path}")

        # Append all pages to the combined output (if more than one, include all)
        for page in reader.pages:
            merger.add_page(page)

    # Write combined PDF
    combined_pdf.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_pdf, "wb") as f:
        merger.write(f)

    print(f"✅ Combined PDF saved to: {combined_pdf} ({len(merger.pages)} pages)")
    return combined_pdf


def main():
    # Usage:
    #   convert_parts_combine.py <base_name> [root_dir] [lool_url] [endpoint]
    #
    # Example:
    #   convert_parts_combine.py ChuyenTienDi_DoanhNghiep
    #   convert_parts_combine.py ChuyenTienDi_DoanhNghiep /Users/me/project/split http://localhost:9980 /lool/convert-to/pdf
    #
    # It will:
    #   - Read sheets from: <root_dir>/<base_name>/sheet/*.xlsx
    #   - Write PDFs to:    <root_dir>/<base_name>/pdf/*.pdf and combined_<base_name>.pdf
    if len(sys.argv) < 2:
        print("Usage: convert_parts_combine.py <base_name> [root_dir] [lool_url] [endpoint]")
        sys.exit(1)

    base_name = sys.argv[1]
    input_root = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("split")
    lool_url = sys.argv[3] if len(sys.argv) >= 4 else "http://localhost:9980"
    endpoint = sys.argv[4] if len(sys.argv) >= 5 else "/lool/convert-to/pdf"

    base_dir = input_root / base_name
    sheet_dir = base_dir / "sheets"
    pdf_dir = base_dir / "pdf"
    combined_pdf = pdf_dir / f"combined_{base_name}.pdf"

    print(f"Base name: {base_name}")
    print(f"Sheet directory: {sheet_dir}")
    print(f"PDF directory: {pdf_dir}")

    if not sheet_dir.exists():
        print(f"❌ Sheet directory not found: {sheet_dir}")
        sys.exit(1)

    sheets = find_sheet_excels(sheet_dir)
    print(f"Found {len(sheets)} sheet files")
    for p in sheets:
        print(f" • {p.name}")

    convert_sheets_and_combine(
        sheet_excels=sheets,
        pdf_dir=pdf_dir,
        combined_pdf=combined_pdf,
        lool_url=lool_url,
        endpoint=endpoint
    )


if __name__ == "__main__":
    main()
