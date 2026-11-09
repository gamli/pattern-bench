uv run --locked --group packaging python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --console `
    --noupx `
    --name pattern-bench `
    --paths src `
    --add-data "src/scanner/default-calibration.ini:scanner" `
    src/main.py
