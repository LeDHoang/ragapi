#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pure-Python XLSX splitter that preserves SmartArt, images, and drawings.

The implementation works directly with the Open Packaging Convention (OPC)
container behind .xlsx files. We keep only the parts required for each sheet
group and recursively pull in their dependent relationships so the resulting
workbooks stay faithful to the source while remaining much smaller.

Outputs are written to:
    split/<workbook name>/sheets/<workbook name>_part_##.xlsx
"""

import argparse
import io
import re
import posixpath
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import xml.etree.ElementTree as ET

from Convert_excel_pdf.config import MAX_SHEETS_PER_SPLIT, SPLIT_OUTPUT_ROOT


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
PROPS_NS = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
VT_NS = "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"
WORKBOOK_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"
)

SKIP_PARTS = {
    "xl/calcChain.xml",
    "xl/_rels/calcChain.xml.rels",
}
SKIP_PREFIXES = (
    "xl/externalLinks",
    "xl/_rels/externalLinks",
    "xl/pivotCache",
    "xl/slicerCaches",
    "xl/timelineCache",
)


@dataclass
class SheetInfo:
    name: str
    rid: str
    target: str
    index: int  # zero-based position in the original workbook


def register_all_namespaces(xml_bytes: bytes) -> None:
    """Preserve original namespace prefixes when rewriting XML payloads."""
    for _, (prefix, uri) in ET.iterparse(io.BytesIO(xml_bytes), events=("start-ns",)):
        ET.register_namespace(prefix or "", uri)


def serialize_xml(root: ET.Element, standalone: Optional[str] = "yes") -> bytes:
    """Serialize an XML element with declaration preserved."""
    buf = io.BytesIO()
    tree = ET.ElementTree(root)
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    xml_bytes = buf.getvalue()
    if standalone and xml_bytes.startswith(b"<?xml"):
        marker = b"?>"
        if marker in xml_bytes[:100]:
            replacement = f' standalone="{standalone}"?>'.encode("utf-8")
            xml_bytes = xml_bytes.replace(marker, replacement, 1)
    return xml_bytes


def normalize_target_path(base_path: str, target: str) -> Optional[str]:
    """Resolve a relationship target against its source part."""
    if not target:
        return None
    fragment_split = target.split("#", 1)
    target_path = fragment_split[0]
    if not target_path:
        return None
    if "://" in target_path:
        return None
    if target_path.startswith("/"):
        resolved = target_path.lstrip("/")
    else:
        base_dir = "" if base_path == "_rels/.rels" else posixpath.dirname(base_path)
        combined = posixpath.join(base_dir, target_path)
        resolved = posixpath.normpath(combined)
    if resolved.startswith("../") or resolved == "..":
        return None
    if resolved.startswith("./"):
        resolved = resolved[2:]
    return resolved


def relationship_part_path(part_path: str) -> str:
    """Return the companion .rels path for a package part."""
    directory, filename = posixpath.split(part_path)
    return posixpath.join(directory, "_rels", f"{filename}.rels") if directory else posixpath.join("_rels", f"{filename}.rels")


def should_skip_part(part_path: str) -> bool:
    """Avoid copying heavy workbook-level caches that Excel will regenerate."""
    path = part_path.lstrip("/")
    if path in SKIP_PARTS:
        return True
    return any(path.startswith(prefix) for prefix in SKIP_PREFIXES)


def write_entry(entries: Dict[str, bytes], written: Set[str], path: str, data: bytes) -> None:
    """Store a part payload and track it for content-types pruning."""
    key = path.lstrip("/")
    entries[key] = data
    written.add("/" + key)


def copy_part_tree(
    part_path: str,
    src_zip: zipfile.ZipFile,
    entries: Dict[str, bytes],
    written: Set[str],
    seen: Set[str],
    src_parts: Set[str],
) -> None:
    """Copy a part and recurse through its internal relationships."""
    part = part_path.lstrip("/")
    if should_skip_part(part) or part in seen:
        return
    if part not in src_parts:
        raise KeyError(f"Required package part missing: {part}")

    seen.add(part)
    write_entry(entries, written, part, src_zip.read(part))

    rels_path = relationship_part_path(part)
    if rels_path not in src_parts or rels_path in seen:
        return

    rels_bytes = src_zip.read(rels_path)
    register_all_namespaces(rels_bytes)
    rels_root = ET.fromstring(rels_bytes)
    new_root = ET.Element(rels_root.tag, rels_root.attrib)
    next_targets: List[str] = []

    for rel in rels_root.findall(f"{{{PKG_REL_NS}}}Relationship"):
        attrs = rel.attrib.copy()
        target_mode = attrs.get("TargetMode", "")
        target = attrs.get("Target", "")

        if target_mode == "External":
            new_root.append(ET.Element(rel.tag, attrs))
            continue

        normalized = normalize_target_path(part, target)
        keep_rel = True

        if normalized:
            if should_skip_part(normalized):
                keep_rel = False
            elif normalized not in src_parts:
                keep_rel = False

        if keep_rel:
            new_root.append(ET.Element(rel.tag, attrs))
            if normalized:
                next_targets.append(normalized)

    if list(new_root):
        write_entry(entries, written, rels_path, serialize_xml(new_root))
        seen.add(rels_path)

    for target in next_targets:
        copy_part_tree(target, src_zip, entries, written, seen, src_parts)


def sheet_markers(name: str) -> Tuple[str, str]:
    """Return unquoted and quoted sheet markers used in formulas."""
    escaped = name.replace("'", "''")
    return f"{name}!", f"'{escaped}'!"


def extract_sheet_references(formula: str) -> Set[str]:
    """Extract referenced sheet names from a defined-name formula."""
    refs: Set[str] = set()
    if "!" not in formula:
        return refs

    pattern = r"(?:'([^']*(?:''[^']*)*)'|([A-Za-z0-9_\\. ]+))!"
    for quoted, bare in re.findall(pattern, formula):
        candidate = quoted or bare
        if not candidate:
            continue
        name = candidate.replace("''", "'").rstrip()
        if name:
            refs.add(name)
    return refs


def prune_workbook_xml(
    workbook_bytes: bytes,
    keep_infos: List[SheetInfo],
) -> bytes:
    """Keep only the selected sheets and tidy workbook-wide metadata."""
    register_all_namespaces(workbook_bytes)
    root = ET.fromstring(workbook_bytes)

    keep_rids = [info.rid for info in keep_infos]
    keep_names = [info.name for info in keep_infos]
    keep_rid_set = set(keep_rids)
    keep_name_set = set(keep_names)
    index_map = {info.index: idx for idx, info in enumerate(keep_infos)}

    sheets_el = root.find(f"{{{MAIN_NS}}}sheets")
    if sheets_el is None:
        raise ValueError("Workbook has no <sheets> element.")

    for sheet_el in list(sheets_el):
        rid = sheet_el.attrib.get(f"{{{REL_NS}}}id")
        if rid not in keep_rid_set:
            sheets_el.remove(sheet_el)

    for idx, sheet_el in enumerate(sheets_el.findall(f"{{{MAIN_NS}}}sheet"), start=1):
        sheet_el.set("sheetId", str(idx))
        if idx == 1:
            sheet_el.set("tabSelected", "1")
        else:
            sheet_el.attrib.pop("tabSelected", None)

    book_views = root.find(f"{{{MAIN_NS}}}bookViews")
    if book_views is not None:
        for view_el in book_views.findall(f"{{{MAIN_NS}}}workbookView"):
            view_el.set("activeTab", "0")

    defined_names = root.find(f"{{{MAIN_NS}}}definedNames")
    if defined_names is not None:
        for name_el in list(defined_names):
            value = (name_el.text or "")
            local_sheet_id = name_el.get("localSheetId")

            if local_sheet_id is not None:
                try:
                    old_idx = int(local_sheet_id)
                except ValueError:
                    defined_names.remove(name_el)
                    continue
                if old_idx not in index_map:
                    defined_names.remove(name_el)
                    continue
                name_el.set("localSheetId", str(index_map[old_idx]))

            sheet_refs = extract_sheet_references(value)
            if sheet_refs and not sheet_refs.issubset(keep_name_set):
                defined_names.remove(name_el)
                continue

        if len(list(defined_names)) == 0:
            root.remove(defined_names)

    return serialize_xml(root)


def prune_workbook_rels(
    workbook_rels_bytes: bytes,
    workbook_path: str,
    keep_rids: List[str],
) -> Tuple[bytes, List[str]]:
    """Drop relationships to removed sheets and hazardous caches."""
    register_all_namespaces(workbook_rels_bytes)
    root = ET.fromstring(workbook_rels_bytes)
    new_root = ET.Element(root.tag, root.attrib)
    keep_rid_set = set(keep_rids)
    extra_targets: List[str] = []

    for rel in root.findall(f"{{{PKG_REL_NS}}}Relationship"):
        attrs = rel.attrib.copy()
        rid = attrs.get("Id", "")
        target = attrs.get("Target", "")
        target_mode = attrs.get("TargetMode", "")
        normalized = normalize_target_path(workbook_path, target)

        if target_mode == "External":
            continue
        if normalized and should_skip_part(normalized):
            continue
        if "externalLinks/" in target or "pivotCache" in target:
            continue
        if normalized and normalized.endswith("calcChain.xml"):
            continue
        if ("worksheets/" in target or "chartsheets/" in target) and rid not in keep_rid_set:
            continue

        new_root.append(ET.Element(rel.tag, attrs))

        if normalized and not (
            normalized.startswith("xl/worksheets") or normalized.startswith("xl/chartsheets")
        ):
            extra_targets.append(normalized)

    return serialize_xml(new_root), extra_targets


def prune_content_types(content_types_bytes: bytes, written_parts: Set[str]) -> bytes:
    """Remove content-type overrides for parts that are no longer present."""
    register_all_namespaces(content_types_bytes)
    root = ET.fromstring(content_types_bytes)

    for override in list(root.findall(f"{{{CONTENT_TYPES_NS}}}Override")):
        part_name = override.attrib.get("PartName", "").lstrip("/")
        if part_name and ("/" + part_name) not in written_parts:
            root.remove(override)

    return serialize_xml(root)


def update_docprops_app(app_bytes: bytes, sheet_names: List[str]) -> bytes:
    """Refresh extended properties to reflect the sheets present in the chunk."""
    if not sheet_names:
        return app_bytes

    register_all_namespaces(app_bytes)
    root = ET.fromstring(app_bytes)

    heading_pairs = root.find(f"{{{PROPS_NS}}}HeadingPairs")
    if heading_pairs is not None:
        vector = heading_pairs.find(f"{{{VT_NS}}}vector")
        if vector is not None:
            items = list(vector.findall(f"{{{VT_NS}}}variant"))
            for idx in range(0, len(items), 2):
                label = items[idx].find(f"{{{VT_NS}}}lpstr")
                count_variant = items[idx + 1] if idx + 1 < len(items) else None
                if (
                    label is not None
                    and label.text == "Worksheets"
                    and count_variant is not None
                ):
                    count = count_variant.find(f"{{{VT_NS}}}i4")
                    if count is not None:
                        count.text = str(len(sheet_names))
            vector.set("size", str(len(items)))

    titles = root.find(f"{{{PROPS_NS}}}TitlesOfParts")
    if titles is not None:
        vector = titles.find(f"{{{VT_NS}}}vector")
        if vector is not None:
            attrs = dict(vector.attrib)
            vector[:] = []
            vector.attrib.clear()
            vector.attrib.update(attrs)
            vector.set("size", str(len(sheet_names)))
            for name in sheet_names:
                lp = ET.Element(f"{{{VT_NS}}}lpstr")
                lp.text = name
                vector.append(lp)

    return serialize_xml(root)


def load_workbook_path(content_types_bytes: bytes) -> str:
    """Locate the workbook part through the content-types manifest."""
    register_all_namespaces(content_types_bytes)
    root = ET.fromstring(content_types_bytes)
    for override in root.findall(f"{{{CONTENT_TYPES_NS}}}Override"):
        if override.attrib.get("ContentType") == WORKBOOK_CONTENT_TYPE:
            part = override.attrib.get("PartName")
            if not part:
                continue
            return part.lstrip("/")
    raise ValueError("Workbook part not declared in [Content_Types].xml")


def load_sheet_infos(
    workbook_bytes: bytes,
    workbook_path: str,
    workbook_rels_bytes: bytes,
) -> List[SheetInfo]:
    """Return ordered sheet descriptors from workbook + rels parts."""
    register_all_namespaces(workbook_rels_bytes)
    rels_root = ET.fromstring(workbook_rels_bytes)
    rid_to_target: Dict[str, str] = {}
    for rel in rels_root.findall(f"{{{PKG_REL_NS}}}Relationship"):
        rid = rel.attrib.get("Id")
        target = rel.attrib.get("Target", "")
        normalized = normalize_target_path(workbook_path, target)
        if rid and normalized:
            rid_to_target[rid] = normalized

    register_all_namespaces(workbook_bytes)
    root = ET.fromstring(workbook_bytes)
    sheets_el = root.find(f"{{{MAIN_NS}}}sheets")
    if sheets_el is None:
        raise ValueError("Workbook is missing the <sheets> collection.")

    sheets: List[SheetInfo] = []
    for idx, sheet_el in enumerate(sheets_el.findall(f"{{{MAIN_NS}}}sheet")):
        name = sheet_el.attrib.get("name", "Sheet")
        rid = sheet_el.attrib.get(f"{{{REL_NS}}}id")
        if not rid:
            continue
        target = rid_to_target.get(rid)
        if not target:
            raise ValueError(f"Relationship '{rid}' has no worksheet target.")
        sheets.append(SheetInfo(name=name, rid=rid, target=target, index=idx))

    if not sheets:
        raise ValueError("No worksheets found in workbook.")
    return sheets


def load_root_relationships(root_rels_bytes: Optional[bytes]) -> Tuple[Optional[bytes], List[str]]:
    """Load package-level relationships and collect targets to copy."""
    if root_rels_bytes is None:
        return None, []
    register_all_namespaces(root_rels_bytes)
    root = ET.fromstring(root_rels_bytes)
    new_root = ET.Element(root.tag, root.attrib)
    targets: List[str] = []

    for rel in root.findall(f"{{{PKG_REL_NS}}}Relationship"):
        attrs = rel.attrib.copy()
        target = attrs.get("Target", "")
        normalized = normalize_target_path("_rels/.rels", target)
        new_root.append(ET.Element(rel.tag, attrs))
        if normalized and normalized != "xl/workbook.xml":
            targets.append(normalized)

    return serialize_xml(new_root), targets


def chunked(iterable: List[SheetInfo], size: int) -> Iterable[List[SheetInfo]]:
    """Yield fixed-size chunks from the sheet list."""
    for idx in range(0, len(iterable), size):
        yield iterable[idx : idx + size]


def split_xlsx(
    input_path: Path,
    output_root: Optional[Path] = None,
    max_sheets: Optional[int] = None,
) -> List[Path]:
    """Split an XLSX into smaller workbooks capped at max_sheets each."""
    if output_root is None:
        output_root = SPLIT_OUTPUT_ROOT
    if max_sheets is None:
        max_sheets = MAX_SHEETS_PER_SPLIT

    if max_sheets < 1:
        raise ValueError("max_sheets must be at least 1")
    if not input_path.exists():
        raise FileNotFoundError(f"Input workbook not found: {input_path}")

    dest_dir = output_root / input_path.stem / "sheets"
    dest_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[Path] = []

    with zipfile.ZipFile(input_path, "r") as src_zip:
        src_parts = set(src_zip.namelist())
        if "[Content_Types].xml" not in src_parts:
            raise ValueError("[Content_Types].xml missing from workbook package.")

        content_types_bytes = src_zip.read("[Content_Types].xml")
        workbook_path = load_workbook_path(content_types_bytes)
        if workbook_path not in src_parts:
            raise ValueError(f"Workbook part '{workbook_path}' missing from archive.")

        workbook_bytes = src_zip.read(workbook_path)
        workbook_rels_path = relationship_part_path(workbook_path)
        if workbook_rels_path not in src_parts:
            raise ValueError(f"Workbook relationships '{workbook_rels_path}' missing.")
        workbook_rels_bytes = src_zip.read(workbook_rels_path)

        sheets = load_sheet_infos(workbook_bytes, workbook_path, workbook_rels_bytes)
        total = len(sheets)
        total_parts = (total + max_sheets - 1) // max_sheets

        root_rels_bytes = src_zip.read("_rels/.rels") if "_rels/.rels" in src_parts else None
        root_rels_serialized, root_targets = load_root_relationships(root_rels_bytes)

        for part_idx, chunk in enumerate(chunked(sheets, max_sheets), start=1):
            entries: OrderedDict[str, bytes] = OrderedDict()
            written: Set[str] = set()
            seen: Set[str] = set()

            if root_rels_serialized is not None:
                write_entry(entries, written, "_rels/.rels", root_rels_serialized)

            for target in root_targets:
                copy_part_tree(target, src_zip, entries, written, seen, src_parts)

            keep_rids = [sheet.rid for sheet in chunk]
            keep_names = [sheet.name for sheet in chunk]

            pruned_workbook = prune_workbook_xml(workbook_bytes, chunk)
            write_entry(entries, written, workbook_path, pruned_workbook)

            pruned_rels, workbook_targets = prune_workbook_rels(
                workbook_rels_bytes, workbook_path, keep_rids
            )
            write_entry(entries, written, workbook_rels_path, pruned_rels)

            for target in workbook_targets:
                copy_part_tree(target, src_zip, entries, written, seen, src_parts)

            for sheet in chunk:
                copy_part_tree(sheet.target, src_zip, entries, written, seen, src_parts)

            if "docProps/app.xml" in entries:
                entries["docProps/app.xml"] = update_docprops_app(
                    entries["docProps/app.xml"], keep_names
                )

            pruned_content_types = prune_content_types(content_types_bytes, written)
            write_entry(entries, written, "[Content_Types].xml", pruned_content_types)

            out_path = dest_dir / f"{input_path.stem}_part_{part_idx:02d}.xlsx"
            if out_path.exists():
                out_path.unlink()

            with zipfile.ZipFile(
                out_path,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            ) as dst_zip:
                for name, data in entries.items():
                    dst_zip.writestr(name, data)

            outputs.append(out_path)
            print(
                f"[split] {input_path.name}: wrote part {part_idx}/{total_parts} -> {out_path.name}"
            )

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split an XLSX into smaller workbooks.")
    parser.add_argument("input_file", type=Path, help="Path to the source XLSX file.")
    parser.add_argument(
        "--max-sheets",
        type=int,
        default=MAX_SHEETS_PER_SPLIT,
        help=f"Maximum sheets per output workbook (default: {MAX_SHEETS_PER_SPLIT}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SPLIT_OUTPUT_ROOT,
        help="Root directory for split outputs (default: split).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = split_xlsx(args.input_file, args.output_root, args.max_sheets)
    if outputs:
        print(f"\n✅ Created {len(outputs)} files under {outputs[0].parent}")
    else:
        print("\n⚠️ No output files created.")


if __name__ == "__main__":
    main()
