"""Extract per-layer hidden states from roberta.base.shuffle.n1 (fairseq) to HDF5.

Converts the fairseq checkpoint to HuggingFace RobertaModel in memory (no
fairseq installation required) and extracts hidden states from all 13 layers
(embedding layer + 12 transformer layers).

Output HDF5 format:
  - key: str(sentence_index)  (0, 1, 2, ...)
  - value: float32 array of shape (13, n_subword_tokens, 768)
  - n_subword_tokens does NOT include BOS/EOS special tokens

This matches the format of the existing RoBERTa embeddings in
/path/to/project-data/Embeddings/RoBERTaBase/.

Usage:
    python scripts/convert_raw_to_roberta_shufflen1.py \\
        --model-dir /path/to/roberta.base.shuffle.n1 \\
        --input     /path/to/ptb3-wsj-train-raw.txt \\
        --output    /path/to/raw.train.roberta-shufflen1-layers.hdf5
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import h5py
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--model-dir', required=True,
    help='Directory containing model.pt (and optionally encoder.json/vocab.bpe).')
parser.add_argument('--conllx', default=None,
                    help='CoNLL-X file; sentences are de-PTBified with '
                         'data.natural_sentence and stored WITH <s>/</s>, matching '
                         'the unified alignment path (scripts/precompute_alignments_hf.py). '
                         'Preferred over --input.')
parser.add_argument('--input', required=False,
    help='Raw text file: one space-tokenised sentence per line.')
parser.add_argument('--output', required=True,
    help='Output HDF5 path.')
args = parser.parse_args()

LAYER_COUNT  = 12   # transformer layers (RoBERTa base)
FEATURE_COUNT = 768
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ---------------------------------------------------------------------------
# Weight mapping: fairseq sentence_encoder → HuggingFace RobertaModel
# ---------------------------------------------------------------------------

def fairseq_to_hf(fairseq_state: dict) -> dict:
    """Remaps a fairseq RoBERTa state-dict to HuggingFace RobertaModel format.

    Fairseq keys use the prefix ``encoder.sentence_encoder.*``.
    HuggingFace uses  ``roberta.*``.
    """
    # Strip the 'model.' prefix that fairseq wraps around everything
    src = {}
    for k, v in fairseq_state.items():
        key = k
        if key.startswith('model.encoder.'):
            key = key[len('model.'):]      # → encoder.*
        elif key.startswith('encoder.'):
            pass                           # already encoder.*
        src[key] = v

    hf = {}

    def copy(src_key, dst_key):
        if src_key in src:
            hf[dst_key] = src[src_key]

    # Embeddings
    copy('encoder.sentence_encoder.embed_tokens.weight',
         'roberta.embeddings.word_embeddings.weight')
    copy('encoder.sentence_encoder.embed_positions.weight',
         'roberta.embeddings.position_embeddings.weight')
    copy('encoder.sentence_encoder.emb_layer_norm.weight',
         'roberta.embeddings.LayerNorm.weight')
    copy('encoder.sentence_encoder.emb_layer_norm.bias',
         'roberta.embeddings.LayerNorm.bias')

    # token_type_embeddings: fairseq RoBERTa doesn't use them; HF needs it
    # Initialise to zeros (shape: 1 × 768)
    hf['roberta.embeddings.token_type_embeddings.weight'] = \
        torch.zeros(1, FEATURE_COUNT)

    # Transformer layers
    for n in range(LAYER_COUNT):
        fs = f'encoder.sentence_encoder.layers.{n}'
        hf_l = f'roberta.encoder.layer.{n}'

        # Self-attention
        copy(f'{fs}.self_attn.q_proj.weight', f'{hf_l}.attention.self.query.weight')
        copy(f'{fs}.self_attn.q_proj.bias',   f'{hf_l}.attention.self.query.bias')
        copy(f'{fs}.self_attn.k_proj.weight', f'{hf_l}.attention.self.key.weight')
        copy(f'{fs}.self_attn.k_proj.bias',   f'{hf_l}.attention.self.key.bias')
        copy(f'{fs}.self_attn.v_proj.weight', f'{hf_l}.attention.self.value.weight')
        copy(f'{fs}.self_attn.v_proj.bias',   f'{hf_l}.attention.self.value.bias')
        copy(f'{fs}.self_attn.out_proj.weight', f'{hf_l}.attention.output.dense.weight')
        copy(f'{fs}.self_attn.out_proj.bias',   f'{hf_l}.attention.output.dense.bias')
        copy(f'{fs}.self_attn_layer_norm.weight', f'{hf_l}.attention.output.LayerNorm.weight')
        copy(f'{fs}.self_attn_layer_norm.bias',   f'{hf_l}.attention.output.LayerNorm.bias')

        # FFN
        copy(f'{fs}.fc1.weight', f'{hf_l}.intermediate.dense.weight')
        copy(f'{fs}.fc1.bias',   f'{hf_l}.intermediate.dense.bias')
        copy(f'{fs}.fc2.weight', f'{hf_l}.output.dense.weight')
        copy(f'{fs}.fc2.bias',   f'{hf_l}.output.dense.bias')
        copy(f'{fs}.final_layer_norm.weight', f'{hf_l}.output.LayerNorm.weight')
        copy(f'{fs}.final_layer_norm.bias',   f'{hf_l}.output.LayerNorm.bias')

    return hf


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print(f'Loading fairseq checkpoint from {args.model_dir} ...')
ckpt_path = Path(args.model_dir) / 'model.pt'
# weights_only=False: PyTorch >=2.6 defaults to True, which refuses this
# fairseq checkpoint because it pickles an omegaconf DictConfig alongside the
# tensors. The file is the checkpoint we downloaded from Facebook Research.
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

fairseq_state = ckpt.get('model', ckpt)   # some checkpoints nest under 'model'
hf_state = fairseq_to_hf(fairseq_state)

# Build model directly on device and load weights in place
config = RobertaConfig.from_pretrained('roberta-base', output_hidden_states=True)
model = RobertaModel(config).to(device)

missing, unexpected = model.load_state_dict(hf_state, strict=False)
print(f'  Missing keys  : {missing}')
print(f'  Unexpected keys: {unexpected}')

model.eval()

# Use the standard roberta-base tokenizer (same GPT-2 BPE vocabulary)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
print('Tokenizer loaded.')


# ---------------------------------------------------------------------------
# Extract and write embeddings
# ---------------------------------------------------------------------------
if args.conllx:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', 'structural-probes'))
    from data import natural_sentence as _natural

    def _read_conllx(path):
        sents, buf = [], []
        for ln in open(path):
            ln = ln.strip()
            if ln.startswith('#'):
                continue
            if not ln:
                if buf:
                    sents.append(buf); buf = []
            else:
                buf.append(ln.split('\t')[1])
        if buf:
            sents.append(buf)
        return sents

    lines = [_natural(t) for t in _read_conllx(args.conllx)]
    USE_SPECIALS = True
    print(f'Read {len(lines)} sentences from {args.conllx} (natural form, with specials)')
elif args.input:
    lines = Path(args.input).read_text().splitlines()
    USE_SPECIALS = False
else:
    parser.error('one of --conllx (preferred) or --input is required')
total = len(lines)
print(f'Processing {total} sentences → {args.output}')

Path(args.output).parent.mkdir(parents=True, exist_ok=True)

with h5py.File(args.output, 'w') as fout:
    for index, line in tqdm(enumerate(lines), total=total, desc='Extracting'):
        line = line.strip()
        if not line:
            continue

        # With --conllx we store <s> ... </s> so that the pre-aligned path's
        # features[1:-1] slice lines up with the alignment matrix rows.
        # With --input we keep the older no-specials convention.
        tokenized = tokenizer.tokenize(line)
        if USE_SPECIALS:
            tokenized = [tokenizer.bos_token] + tokenized + [tokenizer.eos_token]
        if not tokenized:
            # Fallback: store a single zero vector
            dset = fout.create_dataset(str(index),
                (LAYER_COUNT + 1, 1, FEATURE_COUNT), dtype='float32')
            dset[:, :, :] = 0.0
            continue

        ids = tokenizer.convert_tokens_to_ids(tokenized)
        tokens_tensor = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(tokens_tensor)
            # outputs.hidden_states: tuple of (batch=1, seq_len, 768),
            # length = LAYER_COUNT + 1  (embedding layer + 12 transformer layers)
            hidden_states = outputs.hidden_states

        n_toks = len(tokenized)
        dset = fout.create_dataset(str(index),
            (LAYER_COUNT + 1, n_toks, FEATURE_COUNT), dtype='float32')
        dset[:, :, :] = np.vstack(
            [h.squeeze(0).cpu().numpy()[np.newaxis] for h in hidden_states]
        )

    # Same provenance attributes as the other extractors, so that
    # scripts/check_embeddings.py can verify this file's layout against
    # experiments/paper_runs.yaml rather than inferring it from the checkpoint count.
    fout.attrs['n_checkpoints'] = LAYER_COUNT + 1
    fout.attrs['has_midblock'] = False
    fout.attrs['hf_model_name'] = 'roberta.base.shuffle.n1'
    fout.attrs['layout'] = '0=embed_out,k=post_block'
    fout.attrs['convention'] = 'B:consumed'
    fout.attrs['convention_note'] = ('post-LayerNorm hidden_states; RoBERTa is a '
                                     'post-LN encoder, so conventions A and B coincide')

print(f'\nDone. Written to {args.output}')
