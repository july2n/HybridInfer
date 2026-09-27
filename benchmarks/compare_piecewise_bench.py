"""Compare two bench_piecewise_prefill_paths.py parent JSON reports.

python benchmarks/compare_piecewise_bench.py \
    --before logs/bench/piecewise_before.json \
    --after logs/bench/piecewise_after.json \
    --json-out logs/bench/piecewise_comparison.json

Uses median engine wall times and retains eager P2 as a drift control.
"""
import argparse
import json
from pathlib import Path
from statistics import median


def compare(before, after):
    result = {'workloads': {}, 'memory': {}}
    for mode in ('P2', 'P3'):
        a, b = before['modes'][mode], after['modes'][mode]
        if a['config'] != b['config'] or a['meta']['model'] != b['meta']['model']:
            raise ValueError('Benchmark configuration/model differ: ' + mode)
    for name, workload in before['modes']['P3']['workloads'].items():
        times = {}
        for label, report in (('before', before), ('after', after)):
            for mode in ('P2', 'P3'):
                times[label + '_' + mode] = median(
                    run['wall_s'] for run in report['modes'][mode]['workloads'][name]['runs'])
        old_ratio = times['before_P2'] / times['before_P3']
        new_ratio = times['after_P2'] / times['after_P3']
        result['workloads'][name] = {
            **{name + '_ms': round(value * 1000, 3) for name, value in times.items()},
            'piecewise_latency_reduction_pct': round(
                100 * (1 - times['after_P3'] / times['before_P3']), 2),
            'eager_latency_change_pct': round(
                100 * (times['after_P2'] / times['before_P2'] - 1), 2),
            'before_eager_over_piecewise': round(old_ratio, 4),
            'after_eager_over_piecewise': round(new_ratio, 4),
            'relative_speedup_change_pct': round(100 * (new_ratio / old_ratio - 1), 2),
        }
    for phase, values in before['modes']['P3']['memory'].items():
        new_values = after['modes']['P3']['memory'][phase]
        result['memory'][phase] = {
            name + '_reduction_mib': round(values[name] - new_values[name], 1)
            for name in ('allocated_mib', 'reserved_mib', 'max_allocated_mib')
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True)
    parser.add_argument('--after', required=True)
    parser.add_argument('--json-out')
    args = parser.parse_args()
    before = json.loads(Path(args.before).read_text())
    after = json.loads(Path(args.after).read_text())
    result = compare(before, after)
    output = json.dumps(result, indent=2) + '\n'
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output)
    print(output, end='')


if __name__ == '__main__':
    main()
