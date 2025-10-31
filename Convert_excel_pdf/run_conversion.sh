#!/usr/bin/env bash
set -Eeuo pipefail

# Run the full Excel → PDF pipeline described in howtorun.txt
# - Starts Collabora CODE via docker-compose
# - Ensures Python venv and installs requirements
# - Splits large workbooks into chunks
# - Converts each chunk to PDF and produces a combined PDF
#
# Usage:
#   ./run_conversion.sh                 # run defaults from howtorun.txt
#   ./run_conversion.sh file.xlsx:20    # run for a specific file with chunk size 20
#   ./run_conversion.sh a.xlsx:10 b.xlsx:25

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"

info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*" >&2; }
error() { echo "[ERROR] $*" >&2; exit 1; }

# 1) Start Collabora CODE
info "Starting Collabora CODE (docker-compose up -d)"
if ! command -v docker-compose >/dev/null 2>&1; then
  error "docker-compose not found. Install Docker Desktop or docker compose plugin."
fi
docker-compose up -d

# 2) Ensure Python venv and dependencies
if ! command -v python3 >/dev/null 2>&1; then
  error "python3 not found in PATH"
fi

if [ ! -d "venv" ]; then
  info "Creating Python virtualenv at ./venv"
  python3 -m venv venv
fi

info "Activating virtualenv and installing requirements"
# shellcheck disable=SC1091
source venv/bin/activate
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  warn "requirements.txt not found; proceeding with existing environment"
fi

# 3) Determine work items (file:chunk pairs)
declare -a WORK_ITEMS=()
if [ "$#" -eq 0 ]; then
  # Defaults from howtorun.txt
  WORK_ITEMS+=("first8excel2.xlsx:4")
  WORK_ITEMS+=("ChuyenTienDi_DoanhNghiep.xlsx:20")
else
  for arg in "$@"; do
    if [[ "$arg" != *":"* ]]; then
      error "Invalid argument '$arg'. Expected format: <file.xlsx>:<max_sheets>"
    fi
    WORK_ITEMS+=("$arg")
  done
fi

# 4) Process each workbook
for item in "${WORK_ITEMS[@]}"; do
  FILE_PATH=${item%%:*}
  CHUNK=${item##*:}

  if [ ! -f "$FILE_PATH" ]; then
    error "Input file not found: $FILE_PATH"
  fi

  # Derive base name without extension
  FILE_BASE=$(basename "$FILE_PATH")
  BOOK_NAME=${FILE_BASE%.*}

  info "Splitting '$FILE_PATH' into chunks of $CHUNK sheets"
  python3 excel_splitter.py "$FILE_PATH" --max-sheets "$CHUNK"

  info "Converting chunks and combining PDFs for '$BOOK_NAME'"
  python3 convert_parts_combine.py "$BOOK_NAME"

  OUT_DIR="split/$BOOK_NAME/pdf"
  COMBINED="$OUT_DIR/combined_${BOOK_NAME}.pdf"
  if [ -f "$COMBINED" ]; then
    info "Done: $COMBINED"
  else
    warn "Combined PDF not found at: $COMBINED"
  fi
done

info "All requested workbooks processed."

