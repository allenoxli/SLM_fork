import os
from pathlib import Path
import zipfile
def zip_source(filename):
    folder = Path('.')
    files = filter(
        lambda x: ('_pycache_' not in str(x)),
        [f for d in ['codes', 'run.sh'] for f in folder.glob(f'{d}/**/*')]
    )
    with zipfile.ZipFile(filename, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for entry in files:
            zip_file.write(entry, entry.relative_to(folder))


zip_source('zip_file/codes.zip')