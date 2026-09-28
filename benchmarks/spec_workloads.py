"""Local, reproducible prompts shared by speculative validation and timing."""
from pathlib import Path


def natural_cases():
    code = (Path(__file__).resolve().parents[1] /
            'src/hybridinfer/engine/input_prep.py').read_text()
    observations = '\n'.join(
        f'Observation {i}: group {(i * 7) % 31} recorded {13 + i % 47} samples; '
        f'the measurement changed by {(i * 11) % 53} units after intervention {i % 9}. '
        f'The investigator requested a comparison with observation {max(0, i-3)}.'
        for i in range(96))
    return {
        'natural_en': 'Explain why a database index can speed up reads but slow down writes. '
                      'Compare a range query with a full table scan and give concrete examples.',
        'natural_zh': '请解释 CPU 缓存为什么会影响矩阵遍历的速度。分别讨论按行和按列访问、'
                      '缓存行、空间局部性，并给出一个简单的 Python 示例。',
        'repository_code': 'Review this actual Python source. Explain its data flow and suggest '
                           'one concrete improvement:\n```python\n' + code + '\n```',
        'long_records': 'Summarize these distinct observations. Compare trends, mention limitations, '
                        'and identify three observations that deserve follow-up.\n' + observations,
        'low_match': 'Write a concise travel diary describing five different fictional towns. '
                     'Give each town a distinct name, landscape, meal, and memorable encounter.',
    }


def prepare_prompt(tokenizer, text, limit, normalization):
    ids = tokenizer.encode(text)
    if normalization == 'repeat_and_truncate':
        return (ids * (limit // len(ids) + 1))[:limit]
    return ids[:limit]
