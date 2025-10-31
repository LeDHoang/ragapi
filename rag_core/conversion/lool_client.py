#!/usr/bin/env python3
"""
LibreOffice Online (LOOL) client for converting Excel workbooks to PDF.

Ported for integration inside the RAG API so that ingest and parser flows
can rely on the Collabora CODE/LibreOffice Online FullSheetPreview API.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union

import requests
from pypdf import PdfReader, PdfWriter

logger = logging.getLogger(__name__)


class LoolClient:
    """
    Thin wrapper around the LibreOffice Online FullSheetPreview endpoint.

    The service converts an entire workbook in one POST request, returning a
    PDF with one page per sheet. We provide light retry handling so callers
    can recover from transient failures.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:9980",
        endpoint_path: str = "/lool/convert-to/pdf",
        timeout: int = 600,
        max_concurrent: int = 2,
        retry_attempts: int = 3,
        retry_delay: float = 5.0,
    ):
        """
        Args:
            base_url: Base URL of the Collabora/LibreOffice Online service.
            endpoint_path: Endpoint for conversion (e.g. /lool/convert-to/pdf or /cool/...).
            timeout: HTTP timeout in seconds for each conversion request.
            max_concurrent: Reserved for future throttling (not enforced here).
            retry_attempts: Number of attempts before bubbling the exception.
            retry_delay: Seconds to wait between retry attempts.
        """
        self.base_url = base_url.rstrip("/")
        self.endpoint_path = endpoint_path
        self.url = f"{self.base_url}{self.endpoint_path}"
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        self.session = requests.Session()
        # requests.Session ignores unknown attributes, but we store it for clarity.
        self.session.timeout = timeout  # type: ignore[attr-defined]

        logger.info("Initialized LOOL client: %s", self.url)

    def convert_sheet(
        self,
        xlsx_data: Union[bytes, io.BytesIO, Path],
        sheet_index: Optional[int] = None,
        output_path: Optional[Path] = None,
    ) -> bytes:
        """
        Convert a workbook and return either the full PDF or a single page.

        FullSheetPreview always renders every sheet, so we convert once and
        slice the requested page locally if sheet_index is provided.
        """
        all_pages_pdf = self.convert_all_sheets(xlsx_data)

        if sheet_index is None:
            pdf_bytes = all_pages_pdf
        else:
            reader = PdfReader(io.BytesIO(all_pages_pdf))
            if sheet_index < 1 or sheet_index > len(reader.pages):
                raise ValueError(f"Sheet index {sheet_index} outside 1-{len(reader.pages)}")
            writer = PdfWriter()
            writer.add_page(reader.pages[sheet_index - 1])
            output_buffer = io.BytesIO()
            writer.write(output_buffer)
            pdf_bytes = output_buffer.getvalue()

        if output_path:
            output_path.write_bytes(pdf_bytes)
            logger.info("Saved sheet PDF to %s", output_path)

        return pdf_bytes

    @staticmethod
    def count_sheets(xlsx_data: Union[bytes, io.BytesIO, Path]) -> int:
        """Count worksheets by reading xl/workbook.xml from the package."""
        if isinstance(xlsx_data, Path):
            file_data = xlsx_data.read_bytes()
        elif isinstance(xlsx_data, io.BytesIO):
            file_data = xlsx_data.getvalue()
        elif isinstance(xlsx_data, bytes):
            file_data = xlsx_data
        else:
            raise ValueError("xlsx_data must be bytes, BytesIO, or Path")

        try:
            with zipfile.ZipFile(io.BytesIO(file_data)) as zf:
                with zf.open("xl/workbook.xml") as fh:
                    xml_content = fh.read()
                root = ET.fromstring(xml_content)
                ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                sheets = root.findall(".//ns:sheets/ns:sheet", ns)
                sheet_count = len(sheets)
                logger.debug("Workbook contains %d sheets", sheet_count)
                return sheet_count
        except (zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
            raise ValueError(f"Failed to parse workbook.xml: {exc}") from exc

    def convert_all_sheets(
        self,
        xlsx_data: Union[bytes, io.BytesIO, Path],
        output_path: Optional[Path] = None,
        max_workers: Optional[int] = None,  # retained for API parity
    ) -> bytes:
        """
        Convert the provided workbook and return a merged PDF.

        The service responds with one page per worksheet when FullSheetPreview
        is set to true.
        """
        if isinstance(xlsx_data, Path):
            file_data = xlsx_data.read_bytes()
        elif isinstance(xlsx_data, io.BytesIO):
            file_data = xlsx_data.getvalue()
        elif isinstance(xlsx_data, bytes):
            file_data = xlsx_data
        else:
            raise ValueError("xlsx_data must be bytes, BytesIO, or Path")

        sheet_count = self.count_sheets(file_data)
        if sheet_count == 0:
            raise ValueError("No worksheets found in workbook")

        files = {
            "data": (
                "document.xlsx",
                io.BytesIO(file_data),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        }
        data = {"FullSheetPreview": "true"}

        last_exc: Optional[Exception] = None
        for attempt in range(self.retry_attempts):
            try:
                logger.debug("Posting workbook to LOOL (%s attempt %d)", self.url, attempt + 1)
                response = self.session.post(
                    self.url,
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )
                response.raise_for_status()

                pdf_bytes = response.content
                reader = PdfReader(io.BytesIO(pdf_bytes))
                actual_pages = len(reader.pages)
                if actual_pages != sheet_count:
                    logger.warning(
                        "Expected %d pages from LOOL but received %d",
                        sheet_count,
                        actual_pages,
                    )

                lool_headers = {
                    k: v for k, v in response.headers.items() if k.lower().startswith("x-lool")
                }
                logger.info(
                    "[LOOL] Conversion response status=%s pages=%s size=%dB headers=%s",
                    response.status_code,
                    actual_pages,
                    len(pdf_bytes),
                    lool_headers,
                )

                if output_path:
                    output_path.write_bytes(pdf_bytes)
                    logger.info("Saved workbook PDF to %s", output_path)

                return pdf_bytes

            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.retry_attempts - 1:
                    logger.warning(
                        "LOOL conversion attempt %d failed (%s). Retrying in %.1fs",
                        attempt + 1,
                        exc,
                        self.retry_delay,
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.error("LOOL conversion failed after %d attempts", self.retry_attempts)

        assert last_exc is not None  # for mypy
        raise last_exc

    def convert_to_pdf(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        merge_sheets: bool = True,
    ) -> bytes:
        """High-level wrapper kept for backwards compatibility."""
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if merge_sheets:
            return self.convert_all_sheets(input_path, output_path)

        return self.convert_sheet(input_path, output_path=output_path)
