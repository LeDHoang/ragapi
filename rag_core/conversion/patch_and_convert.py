# file: xlsx2pdf_normalize.py
import zipfile, tempfile, os, shutil, subprocess, math
from pathlib import Path
import xml.etree.ElementTree as ET

NS = {
    "s":  "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r":  "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "xdr":"http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing",
    "rels":"http://schemas.openxmlformats.org/package/2006/relationships",
}

A4_PAPER = "9"  # A4. Use "1" for Letter.

def a1(col, row):
    # 1-based col,row -> A1
    name = ""
    while col:
        col, rem = divmod(col-1, 26)
        name = chr(65+rem) + name
    return f"{name}{row}"

def merge_bbox(b1, b2):
    if not b1: return b2
    if not b2: return b1
    (c1,r1,c2,r2) = b1
    (C1,R1,C2,R2) = b2
    return (min(c1,C1), min(r1,R1), max(c2,C2), max(r2,R2))

def dim_to_bbox(ref):
    # "A1:C10" -> (1,1,3,10); "A1" -> (1,1,1,1)
    if ":" in ref:
        a,b = ref.split(":")
    else:
        a=b=ref
    def parse(cell):
        i=0
        while i<len(cell) and cell[i].isalpha(): i+=1
        col_str, row_str = cell[:i], cell[i:]
        col=0
        for ch in col_str.upper():
            col = col*26 + (ord(ch)-64)
        row=int(row_str)
        return col,row
    c1,r1 = parse(a.replace("$",""))
    c2,r2 = parse(b.replace("$",""))
    return (min(c1,c2), min(r1,r2), max(c1,c2), max(r1,r2))

def emu_to_cols_rows(cx, cy, default_col_emu=609600, default_row_emu=190500):
    # Rough: default column ~48pt -> 48*12700 = 609,600 EMU; default row 15pt -> 190,500 EMU
    cols = max(1, math.ceil(cx / default_col_emu))
    rows = max(1, math.ceil(cy / default_row_emu))
    return cols, rows

def read_xml(path):
    t = ET.parse(path)
    return t, t.getroot()

def write_xml(tree, path):
    tree.write(path, encoding="utf-8", xml_declaration=True)

def collect_sheet_map(tmp):
    # Map sheet xml path -> localSheetId (0-based) and name
    wb_tree, wb = read_xml(Path(tmp,"xl","workbook.xml"))
    rels_tree, rels = read_xml(Path(tmp,"xl","_rels","workbook.xml.rels"))
    rid_to_target = {r.get("Id"): r.get("Target") for r in rels.findall("rels:Relationship", NS)}
    sheets = wb.find("s:sheets", NS)
    mapping = {}
    for idx, sh in enumerate(sheets.findall("s:sheet", NS)):
        rid = sh.get(f"{{{NS['r']}}}id")
        target = rid_to_target.get(rid)
        if target and target.endswith(".xml") and "worksheets" in target:
            mapping[Path("xl",target).as_posix()] = {"localSheetId": idx, "name": sh.get("name")}
    return wb_tree, wb, mapping

def remove_defined_names(wb, names=("Print_Area","Print_Titles")):
    dn = wb.find("s:definedNames", NS)
    if dn is None: return
    for n in list(dn):
        if n.get("name","").endswith(tuple(names)) or n.get("name","") in [f"_xlnm.{x}" for x in names]:
            dn.remove(n)
    # remove empty container
    if len(dn)==0:
        wb.remove(dn)

def add_defined_name(wb, localSheetId, name, ref):
    dn = wb.find("s:definedNames", NS)
    if dn is None:
        dn = ET.SubElement(wb, f"{{{NS['s']}}}definedNames")
    el = ET.SubElement(dn, f"{{{NS['s']}}}definedName", {
        "name": f"_xlnm.{name}",
        "localSheetId": str(localSheetId),
    })
    el.text = ref

def sheet_drawings_bbox(tmp, sheet_xml_path):
    # Find drawing rels -> drawing xml -> anchors -> bbox
    rels_path = Path(tmp, sheet_xml_path.replace("worksheets/","worksheets/_rels/") + ".rels")
    if not rels_path.exists(): return None
    try:
        rels_tree, rels = read_xml(rels_path)
    except Exception:
        return None
    targets = [r.get("Target") for r in rels.findall("rels:Relationship", NS) if r.get("Type","").endswith("/drawing")]
    bbox = None
    for tgt in targets:
        draw_path = Path(tmp, "xl", tgt).resolve()
        if not draw_path.exists(): continue
        try:
            d_tree, d = read_xml(draw_path)
        except Exception:
            continue
        # twoCellAnchor
        for anc in d.findall("xdr:twoCellAnchor", NS):
            f = anc.find("xdr:from", NS); t = anc.find("xdr:to", NS)
            if f is None or t is None: continue
            c1 = int(f.findtext("xdr:col", default="0", namespaces=NS)) + 1
            r1 = int(f.findtext("xdr:row", default="0", namespaces=NS)) + 1
            c2 = int(t.findtext("xdr:col", default=str(c1-1), namespaces=NS)) + 1
            r2 = int(t.findtext("xdr:row", default=str(r1-1), namespaces=NS)) + 1
            bbox = merge_bbox(bbox, (min(c1,c2), min(r1,r2), max(c1,c2), max(r1,r2)))
        # oneCellAnchor
        for anc in d.findall("xdr:oneCellAnchor", NS):
            f = anc.find("xdr:from", NS); ext = anc.find("xdr:ext", NS)
            if f is None or ext is None: continue
            c1 = int(f.findtext("xdr:col", default="0", namespaces=NS)) + 1
            r1 = int(f.findtext("xdr:row", default="0", namespaces=NS)) + 1
            cx = int(ext.get("cx","0")); cy = int(ext.get("cy","0"))
            dc, dr = emu_to_cols_rows(cx, cy)
            bbox = merge_bbox(bbox, (c1, r1, c1+dc, r1+dr))
    return bbox

def normalize_sheet_page_setup(sheet_xml_file):
    t, root = read_xml(sheet_xml_file)
    # Remove row/col breaks
    for tag in ("rowBreaks","colBreaks"):
        el = root.find(f"s:{tag}", NS)
        if el is not None:
            root.remove(el)

    # sheetPr / pageSetUpPr fitToPage=1
    sheetPr = root.find("s:sheetPr", NS) or ET.SubElement(root, f"{{{NS['s']}}}sheetPr")
    psup = sheetPr.find("s:pageSetUpPr", NS) or ET.SubElement(sheetPr, f"{{{NS['s']}}}pageSetUpPr")
    psup.set("fitToPage", "1")

    # pageSetup (landscape + 1x1 + A4 + sane margins + no scale)
    ps = root.find("s:pageSetup", NS) or ET.SubElement(root, f"{{{NS['s']}}}pageSetup")
    ps.set("orientation", "landscape")
    ps.set("fitToWidth", "1"); ps.set("fitToHeight", "1")
    if "scale" in ps.attrib: del ps.attrib["scale"]
    ps.set("paperSize", A4_PAPER)
    ps.set("usePrinterDefaults", "0")
    ps.set("horizontalDpi", "300"); ps.set("verticalDpi", "300")
    # margins
    pm = root.find("s:pageMargins", NS)
    if pm is None:
        pm = ET.SubElement(root, f"{{{NS['s']}}}pageMargins",
                           {"left":"0.25","right":"0.25","top":"0.75","bottom":"0.75","header":"0.30","footer":"0.30"})
    write_xml(t, sheet_xml_file)

def set_print_area(tmpdir):
    wb_tree, wb, sheetmap = collect_sheet_map(tmpdir)
    remove_defined_names(wb, ("Print_Area","Print_Titles"))
    for sheet_rel_path, meta in sheetmap.items():
        sheet_xml = Path(tmpdir, sheet_rel_path)
        # 1) normalize page setup
        normalize_sheet_page_setup(sheet_xml)

        # 2) current dimension
        st, root = read_xml(sheet_xml)
        dim = root.find("s:dimension", NS)
        bbox = None
        if dim is not None and dim.get("ref"):
            bbox = dim_to_bbox(dim.get("ref"))

        # 3) include drawings
        dbbox = sheet_drawings_bbox(tmpdir, sheet_rel_path)
        bbox = merge_bbox(bbox, dbbox)

        # 4) fallback bbox if nothing found
        if not bbox:
            bbox = (1,1,1,1)

        # 5) clamp absurd anchors (anti-tiny-page): cap at 1000 cols x 5000 rows
        c1,r1,c2,r2 = bbox
        c2 = min(c2, 1000); r2 = min(r2, 5000)
        if c2 < c1: c2 = c1
        if r2 < r1: r2 = r1

        # Escape single quotes in sheet name (Excel syntax) and build A1 range without backslashes
        safe_name = meta["name"].replace("'", "''")
        ref = "'{}'!{}:{}".format(safe_name, a1(c1, r1), a1(c2, r2))
        add_defined_name(wb, meta["localSheetId"], "Print_Area", ref)

    write_xml(wb_tree, Path(tmpdir,"xl","workbook.xml"))

def patch_xlsx(src, dst):
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(src) as zin:
            zin.extractall(tmp)
        set_print_area(tmp)
        # rezip
        with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for root,_,files in os.walk(tmp):
                for f in files:
                    p = Path(root,f)
                    zout.write(p, os.path.relpath(p, tmp))

def soffice_export(xlsx, pdf_out):
    outdir = str(Path(pdf_out).parent.resolve())
    user_profile = f"file://{tempfile.mkdtemp(prefix='lo_profile_')}"
    # Resolve soffice path (macOS app bundle or PATH)
    soffice = os.environ.get("SOFFICE_BIN") or "soffice"
    if soffice == "soffice" and not shutil.which(soffice):
        mac_path = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
        if Path(mac_path).exists():
            soffice = mac_path
    cmd = [
        soffice,
        f"-env:UserInstallation={user_profile}",
        "--headless","--nologo","--nolockcheck","--norestore","--nofirststartwizard",
        "--convert-to","pdf:calc_pdf_Export",
        "--outdir", outdir,
        str(Path(xlsx).resolve())
    ]
    subprocess.check_call(cmd)
    src_pdf = Path(outdir) / (Path(xlsx).stem + ".pdf")
    if src_pdf != Path(pdf_out):
        shutil.move(src_pdf, pdf_out)

if __name__ == "__main__":
    import sys
    if len(sys.argv)!=3:
        print("Usage: python xlsx2pdf_normalize.py input.xlsx output.pdf"); exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    with tempfile.TemporaryDirectory() as tmp:
        patched = Path(tmp,"patched.xlsx")
        patch_xlsx(src, patched)
        soffice_export(patched, dst)
        print("OK ->", dst)
