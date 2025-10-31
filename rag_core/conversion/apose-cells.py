#need commercial license to use this
import os
import jpype
import jpype.imports

# Set JAVA_HOME if not already set
if 'JAVA_HOME' not in os.environ:
    os.environ['JAVA_HOME'] = '/Users/hoangleduc/Library/Java/JavaVirtualMachines/corretto-21.0.5/Contents/Home'

# Start JVM before importing aspose-cells
if not jpype.isJVMStarted():
    jpype.startJVM()

from asposecells.api import Workbook, PdfSaveOptions, PageOrientationType

wb = Workbook("input/first8excel.xlsx")
for ws in wb.getWorksheets():
    ps = ws.getPageSetup()
    ps.setOrientation(PageOrientationType.LANDSCAPE)
    ps.setFitToPagesWide(1)
    ps.setFitToPagesTall(1)
opts = PdfSaveOptions()
wb.save("input/converted/first8excel.pdf", opts)

# Optionally shutdown JVM when done
jpype.shutdownJVM()
