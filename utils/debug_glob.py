
import glob
import os
import re

root_dir = "/home6/afahad/nobackup27/GFEXPv11.7/GF_OPT72/geosgcm_surf"
print(f"Checking dir: {root_dir}")

# Test Find Last Month Pattern
pat_general = os.path.join(root_dir, "*.geosgcm_surf.*.nc4")
files = glob.glob(pat_general)
print(f"General Pattern found {len(files)} files.")
files.sort()
if files:
    print(f"Last file: {os.path.basename(files[-1])}")
    m = re.search(r'\.(\d{6})\d*.*\.nc4', os.path.basename(files[-1]))
    if m:
        print(f"Parsed Date: {m.group(1)}")
    else:
        print("Regex failed to parse date.")

# Test Specific Date Pattern (2006-01)
year = 2006
month = 1
collection_name = 'geosgcm_surf'
pat_specific = os.path.join(root_dir, f"*.{collection_name}.{year}{month:02d}*.nc4")
print(f"Specific Pattern: {pat_specific}")
files_spec = glob.glob(pat_specific)
print(f"Specific found: {files_spec}")

# Test Permissive Pattern
pat_perm = os.path.join(root_dir, f"*.{collection_name}.*{year}{month:02d}*.nc4")
print(f"Permissive Pattern: {pat_perm}")
files_perm = glob.glob(pat_perm)
print(f"Permissive found: {files_perm}")
