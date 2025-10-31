# lo_fit_and_export.py
import sys, time, os
import uno
from com.sun.star.beans import PropertyValue

def _pv(name, value):
    p = PropertyValue()
    p.Name = name
    p.Value = value
    return p

def file_url(p):
    return uno.systemPathToFileUrl(os.path.abspath(p))

def main(xlsx_path, pdf_path):
    import sys
    import time

    start_time = time.time()
    print(f"[UNO] Starting Excel to PDF conversion: {xlsx_path} → {pdf_path}", file=sys.stderr)

    try:
        # connect to running soffice
        print("[UNO] Connecting to LibreOffice...", file=sys.stderr)
        local = uno.getComponentContext()
        resolver = local.ServiceManager.createInstanceWithContext(
            "com.sun.star.bridge.UnoUrlResolver", local)
        ctx = resolver.resolve("uno:socket,host=127.0.0.1,port=2002;urp;StarOffice.ComponentContext")
        smgr = ctx.ServiceManager
        desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)

        # open spreadsheet hidden
        print("[UNO] Opening Excel file...", file=sys.stderr)
        args = (_pv("Hidden", True),)
        doc = desktop.loadComponentFromURL(file_url(xlsx_path), "_blank", 0, args)

        try:
            # for every sheet, set page style properties
            print("[UNO] Setting page styles for landscape and fit-to-page...", file=sys.stderr)
            styles = doc.getStyleFamilies().getByName("PageStyles")
            sheets = doc.getSheets()
            sheet_count = sheets.getCount()
            print(f"[UNO] Processing {sheet_count} sheets...", file=sys.stderr)

            for i in range(sheet_count):
                sh = sheets.getByIndex(i)
                style_name = sh.PageStyle
                style = styles.getByName(style_name)
                # Landscape and fit to 1x1 pages
                style.setPropertyValue("IsLandscape", True)
                # Fit to width=1, height=1 pages
                style.setPropertyValue("ScaleToPagesX", 1)
                style.setPropertyValue("ScaleToPagesY", 1)
                print(f"[UNO] Sheet {i+1}/{sheet_count}: Set landscape + fit-to-page", file=sys.stderr)

            # export to PDF
            print("[UNO] Exporting to PDF...", file=sys.stderr)
            export_props = (_pv("FilterName", "calc_pdf_Export"),)
            doc.storeToURL(file_url(pdf_path), export_props)

            elapsed = time.time() - start_time
            print(f"[UNO] Conversion completed successfully in {elapsed:.2f}s", file=sys.stderr)

        finally:
            print("[UNO] Closing document...", file=sys.stderr)
            doc.close(True)

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[UNO] ERROR: Conversion failed after {elapsed:.2f}s: {e}", file=sys.stderr)
        raise

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python lo_fit_and_export.py input.xlsx output.pdf")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
