#!/bin/bash
set -e

# Get the directory where this script lives, then cd there
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "📦 Working directory: $(pwd)"
echo "📦 Ensuring data directory exists..."
mkdir -p data

# Download the database ONLY if it doesn't exist
if [ ! -f "data/delivery.db" ]; then
    echo "⬇️ Downloading delivery.db..."
    curl -L -o data/delivery.db "https://techassessment.blob.core.windows.net/aiap24-assessment-data/delivery.db"
else
    echo "✅ delivery.db already exists, skipping download."
fi

echo "Running ML pipeline..."
python3 src/main.py --db_path data/delivery.db
