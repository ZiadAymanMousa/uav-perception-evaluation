# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Validate and render saved test-dev aggregates; optionally derive them from summaries.

This performs no detector inference or AP recomputation. Summary hashes establish
which bytes were read, not independent scientific validation of their contents.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path

METHODS = ('full_640', 'full_960', 'full_1280', 'fixed_slices_640_plus_global')
AP_FIELDS = ('seed', 'method', 'development_ap50_95_percent',
             'testdev_ap50_95_percent', 'testdev_minus_development_APpp')
CONTRAST_FIELDS = ('seed', 'contrast', 'APpp')
COST_FIELDS = ('seed', 'method', 'detector_passes', 'processed_input_pixels',
               'diagnostic_median_ms', 'diagnostic_p95_ms')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    result = float(value)
    require(math.isfinite(result), 'Nonfinite aggregate value')
    return result


def integer(value):
    result = number(value)
    require(result == int(result), 'Noninteger aggregate key/count')
    return int(result)


def parse_csv(raw, fields, label):
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
    require(reader.fieldnames == list(fields), 'Unexpected CSV schema: ' + label)
    rows = list(reader)
    require(all(None not in row and all(v is not None for v in row.values()) for row in rows),
            'Malformed CSV row')
    return rows


def read_csv(path, fields):
    return parse_csv(Path(path).read_bytes(), fields, Path(path).name)


def validate(ap, contrasts, costs):
    expected = {(seed, method) for seed in range(3) for method in METHODS}
    by_key = {}
    for row in ap:
        key = (integer(row['seed']), row['method'])
        require(key in expected and key not in by_key, 'Unexpected/duplicate AP cell')
        dev = number(row['development_ap50_95_percent'])
        test = number(row['testdev_ap50_95_percent'])
        require(0 <= dev <= 100 and 0 <= test <= 100, 'AP outside percent range')
        require(abs(number(row['testdev_minus_development_APpp']) - (test-dev)) < 1e-10,
                'Cross-split difference disagrees with AP')
        by_key[key] = (dev, test)
    require(set(by_key) == expected, 'Missing AP cells')
    seen = set()
    for row in contrasts:
        seed = integer(row['seed'])
        pair = row['contrast'].split(' minus ')
        require(seed in range(3) and len(pair) == 2 and pair[0] == 'full_1280'
                and pair[1] in ('full_960', METHODS[-1]), 'Unexpected contrast')
        key = (seed, pair[1])
        require(key not in seen, 'Duplicate contrast')
        seen.add(key)
        delta = by_key[(seed, 'full_1280')][1] - by_key[(seed, pair[1])][1]
        require(abs(number(row['APpp'])-delta) < 1e-10, 'Contrast disagrees with AP')
    require(len(seen) == 6, 'Missing registered contrast')
    seen = set()
    for row in costs:
        key = (integer(row['seed']), row['method'])
        require(key in expected and key not in seen, 'Unexpected/duplicate cost cell')
        seen.add(key)
        size = {'full_640':640, 'full_960':960, 'full_1280':1280, METHODS[-1]:640}[key[1]]
        passes = integer(row['detector_passes'])
        require(passes == (11448 if key[1] == METHODS[-1] else 1610), 'Unexpected pass count')
        require(integer(row['processed_input_pixels']) == passes * size**2, 'Pixel accounting differs')
        median, p95 = number(row['diagnostic_median_ms']), number(row['diagnostic_p95_ms'])
        require(0 <= median <= p95, 'Invalid diagnostic timing')
    require(seen == expected, 'Missing cost cells')
    return by_key


def from_index(path):
    """Read six explicitly hash-bound summaries. Paths resolve relative to index."""
    path = Path(path).resolve()
    index_bytes = path.read_bytes()
    index = json.loads(index_bytes.decode('utf-8'))
    require(set(index) == {'summaries'} and len(index['summaries']) == 3, 'Expected three summary pairs')
    ap, contrasts, costs, seeds = [], [], [], set()
    hashes = {'summary_index':hashlib.sha256(index_bytes).hexdigest()}
    for pair in index['summaries']:
        require(set(pair) == {'seed','development_summary','development_sha256',
                              'testdev_summary','testdev_sha256'}, 'Unexpected summary-index fields')
        seed = integer(pair['seed'])
        require(seed in range(3) and seed not in seeds, 'Unexpected/duplicate seed')
        seeds.add(seed)
        loaded = {}
        for split in ('development','testdev'):
            source = path.parent / pair[split+'_summary']
            summary_bytes = source.read_bytes()
            actual = hashlib.sha256(summary_bytes).hexdigest()
            require(actual == pair[split+'_sha256'], 'Summary bytes differ from index')
            value = json.loads(summary_bytes.decode('utf-8'))
            require(set(value) == set(METHODS), 'Summary methods missing/extra')
            loaded[split] = value
            hashes[f'seed{seed}_{split}'] = actual
        for method in METHODS:
            dev = loaded['development'][method]['scoring']
            test = loaded['testdev'][method]['scoring']
            require(integer(test['target_count']) == 75102, 'Unexpected test-dev target count')
            require(integer(dev['target_count']) == 38759, 'Unexpected development target count')
            d = 100*number(dev['metrics']['metrics/mAP50-95(B)'])
            t = 100*number(test['metrics']['metrics/mAP50-95(B)'])
            ap.append(dict(zip(AP_FIELDS, (seed,method,d,t,t-d))))
            costs.append({'seed':seed,'method':method,**{key:loaded['testdev'][method][key]
                          for key in COST_FIELDS[2:]}})
        for other in ('full_960', METHODS[-1]):
            full = loaded['testdev']['full_1280']['scoring']['metrics']['metrics/mAP50-95(B)']
            comparison = loaded['testdev'][other]['scoring']['metrics']['metrics/mAP50-95(B)']
            contrasts.append({'seed':seed,'contrast':'full_1280 minus '+other,
                              'APpp':100*number(full)-100*number(comparison)})
    validate(ap, contrasts, costs)
    return ap, contrasts, costs, hashes


def render(ap, contrasts, destination):
    os.environ.setdefault('MPLCONFIGDIR', str(destination / 'matplotlib_config'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios':[1.7,1]})
    labels = ['Full 640','Full 960','Full 1280','Fixed slices\n+ global']
    for seed, colour in zip(range(3), ('#245A81','#A65B14','#33836A')):
        rows = {row['method']:row for row in ap if integer(row['seed']) == seed}
        for split, style in (('development','--'), ('testdev','-')):
            values = [number(rows[m][split+'_ap50_95_percent']) for m in METHODS]
            axes[0].plot(range(4),values,style,marker='o',color=colour,
                         label=f'Seed {seed}, {split}')
        values = [number(next(row['APpp'] for row in contrasts
                  if integer(row['seed']) == seed and row['contrast'] == 'full_1280 minus '+other))
                  for other in ('full_960',METHODS[-1])]
        axes[1].plot([0,1],values,'o-',color=colour,label=f'Seed {seed}')
    axes[0].set_xticks(range(4),labels); axes[0].set_ylabel('Converted AP50–95 (%)')
    axes[0].set_title('Same checkpoints, separate official splits')
    axes[0].legend(fontsize=8,ncol=2)
    axes[1].set_xticks([0,1],['1280 − 960','1280 − slices'])
    axes[1].set_ylabel('Test-dev AP difference (percentage points)')
    axes[1].axhline(0,color='#666666',linewidth=0.8); axes[1].legend(fontsize=8)
    axes[1].set_title('Six declared contrasts')
    for ax in axes:
        ax.grid(axis='y',alpha=.25)
    fig.suptitle('Descriptive comparison; no scene-independence or significance claim',fontsize=11)
    fig.tight_layout()
    for ext in ('png','svg'):
        fig.savefig(destination/f'testdev_development_comparison.{ext}',dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input-dir',type=Path)
    group.add_argument('--summary-index',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), 'Preserve existing output directory')
    if args.summary_index:
        ap, contrasts, costs, hashes = from_index(args.summary_index)
        mode = 'hash-bound-summary derivation; no AP recomputation'
    else:
        specs = [('ap_comparison.csv',AP_FIELDS),('registered_contrasts.csv',CONTRAST_FIELDS),
                 ('diagnostic_costs.csv',COST_FIELDS)]
        buffers = {name:(args.input_dir/name).read_bytes() for name,_ in specs}
        ap, contrasts, costs = [parse_csv(buffers[name],fields,name) for name,fields in specs]
        hashes = {name:hashlib.sha256(buffers[name]).hexdigest() for name,_ in specs}
        validate(ap,contrasts,costs)
        mode = 'saved aggregate rendering; no inference or AP recomputation'
    args.output_dir.mkdir(parents=True)
    for name, fields, rows in [('ap_comparison.csv',AP_FIELDS,ap),
            ('registered_contrasts.csv',CONTRAST_FIELDS,contrasts),('diagnostic_costs.csv',COST_FIELDS,costs)]:
        with (args.output_dir/name).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    render(ap,contrasts,args.output_dir)
    record={'purpose':mode,'input_sha256':hashes,'renderer_sha256':sha(__file__),
            'outputs':{p.name:sha(p) for p in args.output_dir.iterdir() if p.is_file()},
            'limits':['Hashes identify read bytes, not independent verification of their scientific origin',
                      'Same-checkpoint cross-split differences are unpaired descriptive comparisons',
                      'Diagnostic timings do not establish deployment latency or stable speedup']}
    (args.output_dir/'rendering_record.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'rendered','ap_rows':len(ap),'contrasts':len(contrasts),'purpose':mode}))


if __name__ == '__main__':
    main()
