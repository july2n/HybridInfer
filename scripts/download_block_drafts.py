"""Download pinned Qwen3.5-2B block drafts and verify the published LFS hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path

CHECKPOINTS = {
    'dflash': ('taobao-mnn/Qwen3.5-2B-Dflash', '19be0b0fdecfa139728ec6ed97f306f0484f4d50', 'Qwen3.5-2B-DFlash'),
    'dspark': ('rasyosef/Qwen3.5-2B-DSpark', 'bed184c3f0c2abbb058e2b25202be52db6fdfa3b', 'Qwen3.5-2B-DSpark'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['all', *CHECKPOINTS], default='all')
    parser.add_argument('--output-root', default='models')
    parser.add_argument('--endpoint')
    args = parser.parse_args()
    if args.endpoint:
        os.environ['HF_ENDPOINT'] = args.endpoint
    from huggingface_hub import HfApi, snapshot_download
    for method in CHECKPOINTS if args.method == 'all' else [args.method]:
        repo, revision, name = CHECKPOINTS[method]
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        output = Path(args.output_root).expanduser().resolve()/name
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(output), max_workers=2,
                          allow_patterns=['*.safetensors', 'config.json', 'README.md', 'val_metrics.json', 'train_command.txt'])
        weights = []
        for file in info.siblings:
            if not file.rfilename.endswith('.safetensors'):
                continue
            path = output/file.rfilename
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if not file.lfs or actual != file.lfs.sha256 or path.stat().st_size != file.lfs.size:
                raise RuntimeError(f'Published checksum/size mismatch: {path}')
            weights.append(dict(file=file.rfilename, bytes=path.stat().st_size, sha256=actual))
        if not weights:
            raise RuntimeError(f'No checkpoint weights in {repo}')
        manifest = dict(repository=repo, revision=revision, verified=True, weights=weights)
        (output/'download_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
