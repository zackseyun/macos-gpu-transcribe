"""Download pinned public checkpoints; never uploads recordings."""
from pathlib import Path
import json
from huggingface_hub import snapshot_download
ROOT = Path(__file__).resolve().parents[1] / '.mlx-next/models'
MODELS = {
    'granite5': ('ibm-granite/granite-speech-5.0-2b', 'ec49ca6b1a5e7e9176127e09cb303ab78cbacdf7'),
    'nemotron': ('nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-4bit', 'c47153ca2ce4024d43e216b81e6db4f67b886545'),
}
if __name__ == '__main__':
    for name, (repo, revision) in MODELS.items():
        target = ROOT / name
        snapshot_download(repo, revision=revision, local_dir=target,
                          allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])
        (target / 'source.json').write_text(json.dumps({'repo': repo, 'revision': revision}, indent=2))
