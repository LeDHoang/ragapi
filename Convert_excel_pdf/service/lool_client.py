#!/usr/bin/env python3
"""
LibreOffice Online (LOOL) client for converting Excel files to PDF using FullSheetPreview.

This client handles:
- Single sheet conversion with FullSheetPreview
- Multi-sheet workbooks by converting each sheet individually
- Sheet counting via workbook.xml parsing
- PDF merging for multi-sheet output
"""

import io
import logging
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Union

import requests
from pypdf import PdfWriter, PdfReader

logger = logging.getLogger(__name__)


class LoolClient:
    """
    Client for LibreOffice Online FullSheetPreview conversions.

    Handles XLSX/XLS to PDF conversion with one page per worksheet.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:9980",
        endpoint_path: str = "/lool/convert-to/pdf",
        timeout: int = 600,  # 10 minutes for large files
        max_concurrent: int = 2,  # Low concurrency to avoid worker exhaustion
        retry_attempts: int = 3,
        retry_delay: float = 5.0,
    ):
        """
        Initialize the LOOL client.

        Args:
            base_url: Base URL of the LibreOffice Online server
            endpoint_path: API endpoint path (/lool or /cool)
            timeout: HTTP request timeout in seconds
            max_concurrent: Maximum concurrent conversions (kept low)
            retry_attempts: Number of retry attempts on failure
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.endpoint_path = endpoint_path
        self.url = self.base_url + self.endpoint_path
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        # Configure session for connection reuse
        self.session = requests.Session()
        self.session.timeout = timeout

        logger.info(f"Initialized LOOL client: {self.url}")

    def convert_sheet(
        self,
        xlsx_data: Union[bytes, io.BytesIO, Path],
        sheet_index: Optional[int] = None,
        output_path: Optional[Path] = None
    ) -> bytes:
        """
        Convert a single sheet to PDF using FullSheetPreview.

        Note: FullSheetPreview always converts all sheets to separate pages.
        This method converts the entire workbook and extracts the requested sheet's page.

        Args:
            xlsx_data: Excel file data (bytes, BytesIO, or Path)
            sheet_index: 1-based sheet index (None for all sheets)
            output_path: Optional path to save PDF to disk

        Returns:
            PDF bytes (single page if sheet_index specified, all pages otherwise)

        Raises:
            requests.RequestException: On HTTP errors
            ValueError: On invalid input
        """
        # Convert entire workbook first
        all_pages_pdf = self.convert_all_sheets(xlsx_data)

        if sheet_index is None:
            # Return all pages
            pdf_bytes = all_pages_pdf
        else:
            # Extract single page (sheet_index is 1-based, pages are 0-based)
            reader = PdfReader(io.BytesIO(all_pages_pdf))

            if sheet_index < 1 or sheet_index > len(reader.pages):
                raise ValueError(f"Sheet index {sheet_index} is out of range (1-{len(reader.pages)})")

            writer = PdfWriter()
            writer.add_page(reader.pages[sheet_index - 1])
            output_buffer = io.BytesIO()
            writer.write(output_buffer)
            pdf_bytes = output_buffer.getvalue()

        if output_path:
            output_path.write_bytes(pdf_bytes)
            logger.info(f"Saved PDF to {output_path}")

        return pdf_bytes

    @staticmethod
    def count_sheets(xlsx_data: Union[bytes, io.BytesIO, Path]) -> int:
        """
        Count worksheets in an Excel file by parsing workbook.xml.

        Args:
            xlsx_data: Excel file data

        Returns:
            Number of worksheets

        Raises:
            ValueError: If workbook.xml cannot be parsed
        """
        # Read file data if path provided
        if isinstance(xlsx_data, Path):
            with open(xlsx_data, 'rb') as f:
                file_data = f.read()
        elif isinstance(xlsx_data, io.BytesIO):
            file_data = xlsx_data.getvalue()
        elif isinstance(xlsx_data, bytes):
            file_data = xlsx_data
        else:
            raise ValueError("xlsx_data must be bytes, BytesIO, or Path")

        try:
            with zipfile.ZipFile(io.BytesIO(file_data)) as zf:
                with zf.open("xl/workbook.xml") as f:
                    xml_content = f.read()

                # Parse XML with namespace handling
                root = ET.fromstring(xml_content)
                ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

                # Find all sheet elements
                sheets = root.findall(".//ns:sheets/ns:sheet", ns)
                sheet_count = len(sheets)

                logger.debug(f"Found {sheet_count} worksheets")
                return sheet_count

        except (zipfile.BadZipFile, KeyError, ET.ParseError) as e:
            raise ValueError(f"Failed to parse workbook.xml: {e}")

    def convert_all_sheets(
        self,
        xlsx_data: Union[bytes, io.BytesIO, Path],
        output_path: Optional[Path] = None,
        max_workers: Optional[int] = None
    ) -> bytes:
        """
        Convert all sheets to a single PDF using FullSheetPreview.

        FullSheetPreview automatically creates one page per worksheet.

        Args:
            xlsx_data: Excel file data
            output_path: Optional path to save PDF
            max_workers: Ignored (single conversion)

        Returns:
            PDF bytes with one page per worksheet
        """
        # Read file data if path provided
        if isinstance(xlsx_data, Path):
            with open(xlsx_data, 'rb') as f:
                file_data = f.read()
        elif isinstance(xlsx_data, io.BytesIO):
            file_data = xlsx_data.getvalue()
        elif isinstance(xlsx_data, bytes):
            file_data = xlsx_data
        else:
            raise ValueError("xlsx_data must be bytes, BytesIO, or Path")

        # Count sheets first for logging
        sheet_count = self.count_sheets(file_data)
        if sheet_count == 0:
            raise ValueError("No worksheets found in file")

        logger.info(f"Converting {sheet_count} sheets to PDF (single FullSheetPreview call)")

        # Single API call with FullSheetPreview (no Sheet parameter)
        files = {
            "data": ("document.xlsx", io.BytesIO(file_data), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        }
        data = {"FullSheetPreview": "true"}

        # Retry logic
        last_exception = None
        for attempt in range(self.retry_attempts):
            try:
                logger.debug(f"Converting workbook (attempt {attempt + 1})")

                response = self.session.post(
                    self.url,
                    files=files,
                    data=data,
                    timeout=self.timeout
                )
                response.raise_for_status()

                pdf_bytes = response.content

                # Verify we got the expected number of pages
                reader = PdfReader(io.BytesIO(pdf_bytes))
                actual_pages = len(reader.pages)
                if actual_pages != sheet_count:
                    logger.warning(f"Expected {sheet_count} pages but got {actual_pages}")

                if output_path:
                    output_path.write_bytes(pdf_bytes)
                    logger.info(f"Saved PDF to {output_path}")

                return pdf_bytes

            except requests.RequestException as e:
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All {self.retry_attempts} attempts failed")

        raise last_exception

    def convert_to_pdf(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        merge_sheets: bool = True
    ) -> bytes:
        """
        High-level method to convert Excel file to PDF.

        Args:
            input_path: Path to input Excel file
            output_path: Optional path for output PDF
            merge_sheets: If True, merge all sheets into one PDF

        Returns:
            PDF bytes
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if merge_sheets:
            return self.convert_all_sheets(input_path, output_path)
        else:
            # Convert only the active sheet
            return self.convert_sheet(input_path, output_path=output_path)
