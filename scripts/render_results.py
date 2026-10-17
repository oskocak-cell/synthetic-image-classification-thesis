"""Draw a dependency-free SVG from the saved thesis result table; no ML runs."""
import csv
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / 'results/classification_results.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(r['Dataset'], r['Model'], r['Input'].split()[0]): float(r['Macro F1']) for r in rows}
    datasets = ['Fashion-MNIST', 'EuroSAT', 'PathMNIST']
    models = ['CNN', 'RF', 'CNN-head', 'RF-features']
    sources = [('Real', '#254c72'), ('DDIM', '#198f8b'), ('CWGAN-GP', '#cb7838')]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="520" viewBox="0 0 1200 520" role="img" aria-labelledby="title desc">',
           '<title id="title">Macro F1 on real test images</title>',
           '<desc id="desc">All twelve dataset and classifier combinations, comparing real training with DDIM and CWGAN-GP synthetic-only training. Saved thesis results; not independently rerun.</desc>',
           '<rect width="1200" height="520" fill="#f6f8fb"/>',
           '<g font-family="Arial, sans-serif" fill="#182e48">',
           '<text x="30" y="37" font-size="25" font-weight="700">Macro F1 on real test images</text>',
           '<text x="30" y="61" font-size="14">Real-data baselines versus synthetic-only training · Saved thesis results, not a new replication</text>']
    for index, (source, color) in enumerate(sources):
        x = 30 + index * 215
        svg += [f'<rect x="{x}" y="78" width="13" height="13" fill="{color}"/>',
                f'<text x="{x+21}" y="89" font-size="13">{escape(source)} training</text>']
    for index, dataset in enumerate(datasets):
        left = 30 + 390 * index
        svg.append(f'<text x="{left}" y="128" font-size="20" font-weight="700">{escape(dataset)}</text>')
        for tick in [0, .5, 1]:
            x = left + 90 + tick * 210
            svg += [f'<line x1="{x}" y1="155" x2="{x}" y2="450" stroke="#dce3ec"/>',
                    f'<text x="{x}" y="149" text-anchor="middle" font-size="11">{tick:g}</text>']
        for j, model in enumerate(models):
            y = 170 + j * 73
            svg.append(f'<text x="{left}" y="{y+19}" font-size="13">{escape(model)}</text>')
            for k, (source, color) in enumerate(sources):
                value = lookup[(dataset, model, source)]; width = value * 210; top = y + k * 15
                svg += [f'<rect x="{left+90}" y="{top}" width="{width:.2f}" height="10" rx="2" fill="{color}"/>',
                        f'<text x="{left+96+width:.2f}" y="{top+9}" font-size="10">{value:.3f}</text>']
    svg += ['<text x="30" y="485" font-size="13" fill="#50617a">Source: saved classification_results.csv, checked against submitted thesis Table 14 · Higher is better</text>',
            '</g></svg>']
    (ROOT / 'docs/results-overview.svg').write_text('\n'.join(svg) + '\n')
    print('Rendered docs/results-overview.svg from 36 saved comparisons.')


if __name__ == '__main__':
    main()
