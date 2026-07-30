'''Extract ALL residual-stream checkpoints (post-attention and post-block) for any
HuggingFace encoder or decoder, on de-PTBified natural sentences.

`AutoModel(output_hidden_states=True)` only exposes the residual stream after each
FULL transformer block. Hewitt-style analyses of BERT-base in this project also use
the MID-block stream (after the attention sub-layer, before the feed-forward), which
is where BERT-base's optimal checkpoint (16) actually lies. This script recovers
those with forward hooks, reproducing the transformer_lens layout:

    idx 0        raw token embeddings (no position, no LayerNorm)
    idx 1        embedding output (word + position + type + LayerNorm)
    idx 2k       post-attention residual stream of block k   (k = 1..L)
    idx 2k+1     post-block residual stream of block k       (k = 1..L)
    -> 2 + 2L checkpoints total (26 for a 12-block model, matching BERT-base)

Where the mid-block state lives differs by architecture, so the hook strategy is
selected explicitly per model type and unknown types raise rather than guess:

  post_ln  (BERT, RoBERTa, DeBERTa-v2/v3)
      Post-LayerNorm blocks. We capture the residual stream BEFORE each sub-layer's
      LayerNorm, by pre-hooking `attention.output.LayerNorm` and `output.LayerNorm`.
      This is the unnormalized accumulator, matching what transformer_lens exposes
      after folding LayerNorm (verified against the existing BERT-base extraction).
      Reading the LayerNorm *output* instead would give a differently scaled
      representation: same directions, but std ~0.8 rather than ~1.3-1.7.

  pre_ln   (GPT-2, ModernBERT)
      Pre-LayerNorm blocks compute `h = h + attn(LN(h))`. Capture the block input
      with a forward-pre-hook and the attention output with a forward hook, then
      add them.

  none     (GPT-J)
      GPT-J runs attention and MLP IN PARALLEL off the same LayerNormed input:
          h = attn(ln(h)) + mlp(ln(h)) + residual
      The residual stream therefore never holds an attention-only update, and a
      "post-attention checkpoint" does not exist. Such models are extracted with
      post-block checkpoints only (pass --allow-missing-midblock to acknowledge).

Usage:
  python convert_raw_to_hf_all_checkpoints.py IN.conllx OUT.hdf5 bert-base-cased
  python convert_raw_to_hf_all_checkpoints.py IN.conllx OUT.hdf5 EleutherAI/gpt-j-6B \
      --revision float16 --model-dtype float16 --allow-missing-midblock
'''
import os
import sys
from argparse import ArgumentParser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'structural-probes'))

import h5py
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from data import natural_sentence

# model_type -> (strategy, attribute path to the list of blocks)
STRATEGY = {
    'bert':        ('post_ln', 'encoder.layer'),
    'roberta':     ('post_ln', 'encoder.layer'),
    'deberta-v2':  ('post_ln', 'encoder.layer'),
    'gpt2':        ('pre_ln',  'h'),
    'modernbert':  ('pre_ln',  'layers'),
    'gptj':        ('none',    'h'),
}


def resolve(root, path):
    obj = root
    for part in path.split('.'):
        obj = getattr(obj, part)
    return obj


def first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for e in x:
            if torch.is_tensor(e):
                return e
    raise TypeError(f'no tensor in hook output of type {type(x)}')


def read_conllx_sentences(path):
    sentences, buf = [], []
    for line in open(path):
        line = line.strip()
        if line.startswith('#'):
            continue
        if not line:
            if buf:
                sentences.append(buf)
                buf = []
        else:
            buf.append(line.split('\t')[1])
    if buf:
        sentences.append(buf)
    return sentences


def main():
    argp = ArgumentParser()
    argp.add_argument('input_path')
    argp.add_argument('output_path')
    argp.add_argument('model_name')
    argp.add_argument('--revision', default=None)
    argp.add_argument('--model-dtype', default='float32', choices=['float16', 'float32'])
    argp.add_argument('--dtype', default='float32', choices=['float16', 'float32'],
                      help='dtype the checkpoints are stored in')
    argp.add_argument('--allow-missing-midblock', action='store_true',
                      help='required for architectures with no mid-block state (GPT-J)')
    args = argp.parse_args()

    store_dtype = np.float16 if args.dtype == 'float16' else np.float32
    wdtype = torch.float16 if args.model_dtype == 'float16' else torch.float32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.revision)
    model = AutoModel.from_pretrained(args.model_name, output_hidden_states=True,
                                      dtype=wdtype, low_cpu_mem_usage=True,
                                      revision=args.revision).to(device).eval()

    mtype = model.config.model_type
    if mtype not in STRATEGY:
        raise ValueError(
            f"model_type '{mtype}' has no registered mid-block strategy. Add one to "
            f"STRATEGY after checking where that architecture's attention sub-layer "
            f"writes into the residual stream; guessing would silently misalign checkpoints.")
    strategy, blocks_path = STRATEGY[mtype]
    blocks = resolve(model, blocks_path)
    n_layers = len(blocks)

    if strategy == 'none' and not args.allow_missing_midblock:
        raise ValueError(
            f"{args.model_name} ({mtype}) applies attention and the feed-forward in "
            f"parallel, so no post-attention residual stream exists. Re-run with "
            f"--allow-missing-midblock to extract post-block checkpoints only.")

    has_mid = strategy != 'none'
    n_states = 2 + 2 * n_layers if has_mid else 1 + n_layers
    hidden = getattr(model.config, 'hidden_size', None) or model.config.n_embd
    print(f'{args.model_name}: type={mtype} layers={n_layers} hidden={hidden} '
          f'strategy={strategy} -> {n_states} checkpoints, store={args.dtype}')

    # ---- hooks -------------------------------------------------------------
    mid, post = {}, {}
    handles = []
    if strategy == 'post_ln':
        for i, blk in enumerate(blocks):
            handles.append(blk.attention.output.LayerNorm.register_forward_pre_hook(
                lambda m, inp, i=i: mid.__setitem__(i, first_tensor(inp))))
            handles.append(blk.output.LayerNorm.register_forward_pre_hook(
                lambda m, inp, i=i: post.__setitem__(i, first_tensor(inp))))
    if strategy in ('pre_ln', 'none'):
        # hidden_states[-1] has the model's FINAL norm applied (ln_f / final_norm),
        # unlike every other entry. Using it would put the last checkpoint in a
        # different space from its neighbours, so read block outputs directly.
        for i, blk in enumerate(blocks):
            handles.append(blk.register_forward_hook(
                lambda m, inp, out, i=i: post.__setitem__(i, first_tensor(out))))
    if strategy == 'pre_ln':
        blk_in = {}
        for i, blk in enumerate(blocks):
            handles.append(blk.register_forward_pre_hook(
                lambda m, inp, i=i: blk_in.__setitem__(i, first_tensor(inp))))
            attn = getattr(blk, 'attn', None) or getattr(blk, 'attention')
            handles.append(attn.register_forward_hook(
                lambda m, inp, out, i=i: mid.__setitem__(
                    i, blk_in[i] + first_tensor(out))))

    sentences = read_conllx_sentences(args.input_path)
    print(f'Read {len(sentences)} sentences from {args.input_path}')
    embed = model.get_input_embeddings()

    with h5py.File(args.output_path, 'w') as fout:
        for index, tokens in enumerate(sentences):
            mid.clear(); post.clear()
            natural = natural_sentence(tokens)
            enc = tokenizer(natural, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**enc)
                raw = embed(enc['input_ids'])

            hs = out.hidden_states                 # (L+1) x (1, seq, hidden)
            states = []
            if has_mid:
                states.append(raw[0])              # idx 0: raw token embeddings
                states.append(hs[0][0])            # idx 1: embedding output
                for k in range(n_layers):
                    states.append(mid[k][0])       # idx 2k+2: post-attention
                    # post_ln: use the pre-LayerNorm accumulator, not hidden_states,
                    # so both checkpoints live in the same (unnormalized) space.
                    states.append(post[k][0])   # raw residual stream, never final-normed
            else:
                states.append(hs[0][0])
                for k in range(n_layers):
                    states.append(post[k][0])   # raw block output, not hidden_states[-1]

            arr = torch.stack(states, dim=0).float().cpu().numpy().astype(store_dtype)
            assert arr.shape[0] == n_states, (arr.shape, n_states)
            fout.create_dataset(str(index), data=arr)

            if (index + 1) % 500 == 0:
                print(f'Processed {index + 1} sentences', flush=True)

        fout.attrs['n_checkpoints'] = n_states
        fout.attrs['has_midblock'] = bool(has_mid)
        fout.attrs['hf_model_name'] = args.model_name
        fout.attrs['layout'] = ('0=raw_embed,1=embed_out,2k=post_attn,2k+1=post_block'
                                if has_mid else '0=embed_out,k=post_block')

    for h in handles:
        h.remove()
    print('Done.')


if __name__ == '__main__':
    main()
