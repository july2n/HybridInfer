"""Export numerical experiment summary, token CSV and error curves."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    if not data['completed']:
        raise ValueError('Experiment is incomplete')
    summary, tokens = [], []
    for result in data['results']:
        rows, blocks = result['rows'], result['blocks']
        summary.append(dict(variant=result['variant'], mode=result['state_mode'],
            predictions=len(rows), first_flip=result['first_flip'], flips=result['flips'],
            exact_logits=sum(row['logits']['equal'] for row in rows),
            max_logit_error=max(row['logits']['max_abs'] for row in rows),
            max_recurrent_error=max(s['recurrent']['max_abs'] for b in blocks for s in b['states']),
            all_layers_exact=all(s['equal'] for row in rows for s in row['layers']),
            all_states_exact=all(s['conv']['equal'] and s['recurrent']['equal']
                                 for b in blocks for s in b['states']),
            all_kv_exact=all(b['kv']['equal'] for b in blocks)))
        for row in rows:
            tokens.append(dict(variant=result['variant'], mode=result['state_mode'],
                output_token_number=row['output_token_number'], margin=row['margin'],
                logit_max_abs=row['logits']['max_abs'], argmax_equal=row['argmax_equal'],
                directional_perturbation=row['directional_perturbation'],
                contender_reference_gap=row['contender_reference_gap'],
                first_different_layer=next((s['layer'] for s in row['layers'] if not s['equal']), None)))
    stem = args.input.with_suffix('')
    Path(str(stem)+'_summary.json').write_text(json.dumps(summary, indent=2))
    with Path(str(stem)+'_tokens.csv').open('w') as out:
        writer = csv.DictWriter(out, fieldnames=list(tokens[0]))
        writer.writeheader()
        writer.writerows(tokens)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    modes = list(dict.fromkeys(r['state_mode'] for r in data['results']))
    fig, axes = plt.subplots(2, len(modes), figsize=(7*len(modes), 8), squeeze=False)
    for col, mode in enumerate(modes):
        for result in data['results']:
            if result['state_mode'] != mode:
                continue
            rows, blocks = result['rows'], result['blocks']
            axes[0,col].plot([r['output_token_number'] for r in rows],
                [r['logits']['max_abs'] for r in rows], label=result['variant'])
            flips = [r for r in rows if not r['argmax_equal']]
            axes[0,col].scatter([r['output_token_number'] for r in flips],
                [r['logits']['max_abs'] for r in flips], s=22, marker='x')
            axes[1,col].plot([b['input_offset']+b['query_count']+1 for b in blocks],
                [max(s['recurrent']['max_abs'] for s in b['states']) for b in blocks],
                label=result['variant'])
        for row, label in enumerate(('Logit max absolute error (x = flip)',
                                     'Block-end recurrent max absolute error')):
            axes[row,col].set_title(mode+' / '+label)
            axes[row,col].set_xlabel('Generated output token number (1-based)')
            axes[row,col].set_yscale('symlog', linthresh=1e-6)
            axes[row,col].grid(alpha=.25)
        axes[0,col].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(stem)+'_errors.png', dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
