# Excel to PDF Converter using LibreOffice Online

Converts Excel files (.xlsx, .xls) to PDF with one page per worksheet using LibreOffice Online's FullSheetPreview feature. Optimized for large workbooks while preserving SmartArt, images, charts, and complex layouts.

## Features

- ✅ **One page per worksheet**: Each sheet becomes exactly one PDF page
- ✅ **Handles SmartArt & images**: No coordinate calculation needed
- ✅ **Landscape scaling**: Automatic fit-to-page scaling
- ✅ **Headless operation**: No GUI required
- ✅ **Multi-sheet merging**: Combine all sheets into single PDF
- ✅ **Large file support**: Optimized for big Excel files
- ✅ **Docker ready**: Containerized LibreOffice Online setup

## Quick Start

### 1. Start LibreOffice Online (Collabora CODE)

```bash
docker-compose up -d
docker-compose logs -f collabora  # wait for healthy
```

### 2. Set up Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Recommended for large workbooks: split then convert

```bash
# Split into smaller parts (choose a chunk size that avoids timeouts)
python3 excel_splitter.py first8excel2.xlsx --max-sheets 4
python3 excel_splitter.py ChuyenTienDi_DoanhNghiep.xlsx --max-sheets 20

# Convert parts and create combined PDF (venv must be active)
python3 convert_parts_combine.py first8excel2
python3 convert_parts_combine.py ChuyenTienDi_DoanhNghiep
```

Outputs:
- Per-workbook PDFs under `split/<workbook>/pdf/*.pdf`
- Combined file at `split/<workbook>/pdf/combined_<workbook>.pdf`

## Excel Splitting

For Excel files with many sheets, you can split them into smaller Excel files with a maximum number of sheets per file. This preserves full content by operating on the XLSX package parts.

```bash
# Split into files with max 20 sheets each (default)
python excel_splitter.py large_workbook.xlsx

# Split into files with max 10 sheets each
python excel_splitter.py large_workbook.xlsx --max-sheets 10

# Specify output directory
python excel_splitter.py large_workbook.xlsx --max-sheets 5 --output-dir ./split_files
```

Output files are named: `[filename]_part_1.xlsx`, `[filename]_part_2.xlsx`, etc., in `split/<workbook>/sheets/`.

**Note**: This splitter works by treating .xlsx files as ZIP archives and directly manipulating XML content, ensuring 100% preservation of all Excel features including SmartArt, embedded images, complex formatting, and relationships. It completely avoids Excel parsing libraries to prevent corruption and hanging issues.

## Batch Conversion

Two options:
- Split parts then convert and combine (best for very large files):
  - See Quick Start step 3 above (`excel_splitter.py` + `convert_parts_combine.py`).
- Convert in one go and slice pages locally (fine for smaller workbooks):
  - `python3 batch_convert.py workbook.xlsx`
  - Or split first then slice: `python3 batch_convert.py workbook.xlsx --deep`

Output structure (split path):
```
split/<workbook>/
├── sheets/
│   ├── <workbook>_part_01.xlsx
│   └── ...
└── pdf/
    ├── <workbook>_part_01.pdf
    ├── ...
    └── combined_<workbook>.pdf
```

## Advanced Usage

### Custom Server URL

```bash
# Use different LOOL server
LOOL_URL=http://my-collabora-server:9980 python3 batch_convert.py input.xlsx

# For newer CODE versions (22+), use /cool endpoint
python3 batch_convert.py input.xlsx --endpoint /cool/convert-to/pdf
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOOL_URL` | `http://localhost:9980` | LibreOffice Online server URL |
| `LOOL_ENDPOINT` | `/lool/convert-to/pdf` | API endpoint path (`/cool/convert-to/pdf` on newer CODE) |
| `CODE_ADMIN_USER` | `admin` | Admin username for CODE console |
| `CODE_ADMIN_PASS` | `adminadmin` | Admin password for CODE console |

## Docker Setup Details

### Custom Fonts

For better rendering fidelity with corporate fonts:

1. Create a `fonts/` directory
2. Add your `.ttf` or `.otf` files
3. Restart the container:

```bash
docker-compose restart collabora
```

### Production Security

For production deployment, modify `docker-compose.yml`:

```yaml
environment:
  - extra_params=--o:ssl.enable=true --o:ssl.termination=false --o:net.post_allow.host[0]=192.168.1.0/24 --o:storage.wopi.host[0]=your-domain.com
```

## API Usage

You can also use the client programmatically:

```python
from service.lool_client import LoolClient

client = LoolClient()

# Convert all sheets to merged PDF
pdf_bytes = client.convert_all_sheets("workbook.xlsx")

# Convert single sheet
pdf_bytes = client.convert_sheet("workbook.xlsx", sheet_index=2)

# Count sheets
count = client.count_sheets("workbook.xlsx")
```

## Troubleshooting

### Common Issues

1. **Connection refused**: Ensure Collabora container is running and healthy
2. **Timeout errors**: Large files may need longer timeouts (use `--timeout 1200`)
3. **Font rendering issues**: Add missing fonts to the `fonts/` directory
4. **Memory issues**: Ensure Docker has sufficient memory allocation

### Health Check

```bash
# Check if service is ready (wget is present in our healthcheck)
wget -q -O - http://localhost:9980/hosting/discovery >/dev/null

# Quick conversion smoke test
curl -F "data=@first8excel2.xlsx" -F "FullSheetPreview=true" http://localhost:9980/lool/convert-to/pdf > /tmp/out.pdf

# View container logs
docker-compose logs collabora
```

## Architecture

- LibreOffice Online (Collabora CODE): Handles Excel parsing and PDF rendering
- FullSheetPreview API: One page per sheet via HTTP POST
- Python client (`service/lool_client.py`): Orchestrates conversions and merging
- Docker: Isolated, reproducible environment

## Helper Script

You can run the full pipeline with:

```bash
chmod +x run_conversion.sh
./run_conversion.sh                 # uses defaults from howtorun.txt
./run_conversion.sh file.xlsx:20    # custom file and chunk size
```

## License

This project uses Collabora CODE, which is free for personal and small business use. Check Collabora's licensing terms for commercial deployments.
