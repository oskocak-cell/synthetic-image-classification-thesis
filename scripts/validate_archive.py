"""Static archive checks only. Does not run training or replicate thesis metrics."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SENSITIVE = re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}|hf_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|/Users/|/home/|2107437|u507074|aurometalsaurus')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    manifest = json.loads((ROOT / 'docs/source-manifest.json').read_text())
    require(len(manifest) == 182, 'Expected 149 main files and 33 recovered autosaves')
    counts = {'.py': 0, '.ipynb': 0}
    for item in manifest:
        path = ROOT / item['published_path']
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == item['published_file_sha256'], f'Published file changed without manifest update: {item["published_path"]}')
        text = data.decode('utf-8')
        require(not SENSITIVE.search(text), f'Review sensitive pattern in {item["published_path"]}')
        counts[path.suffix] += 1
        if path.suffix == '.py':
            ast.parse(text, filename=str(path))
        else:
            notebook = json.loads(text)
            require(notebook.get('nbformat') == 4, 'Expected notebook format 4')
            codes = [c for c in notebook['cells'] if c['cell_type'] == 'code']
            require(any(''.join(c.get('source', [])).strip() for c in codes), f'Empty notebook: {path.name}')
            require(all(not c.get('outputs') and c.get('execution_count') is None for c in codes), f'Clear notebook outputs before publishing: {path.name}')
    require(counts == {'.py': 17, '.ipynb': 165}, 'Unexpected code counts')
    with (ROOT / 'results/classification_results.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 36, 'Expected 36 saved result comparisons')
    keys = {(r['Dataset'], r['Model'], r['Input'].split()[0]) for r in rows}
    expected = {(d, m, s) for d in ['Fashion-MNIST', 'EuroSAT', 'PathMNIST'] for m in ['CNN', 'RF', 'CNN-head', 'RF-features'] for s in ['Real', 'DDIM', 'CWGAN-GP']}
    require(keys == expected, 'Missing or duplicate result combinations')
    for row in rows:
        for metric in ['Accuracy', 'Macro Prec.', 'Macro Rec.', 'Macro F1']:
            require(0 <= float(row[metric]) <= 1, 'Metric outside [0,1]')
    for path in ROOT.rglob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if not target.startswith(('http:', 'https:', '#')):
                require((path.parent / unquote(target.split('#')[0])).exists(), f'Missing documentation link: {target}')
    ET.parse(ROOT / 'docs/results-overview.svg')
    print('Checked 165 nonempty, output-cleared notebooks, 17 Python scripts, published-file hashes, 36 result rows, documentation links, and SVG structure.')
    print('Static checks only: no ML experiments were executed.')


if __name__ == '__main__':
    main()
