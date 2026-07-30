'''Derive "convention B" checkpoints from "convention A" checkpoints.

Two conventions appear in the literature for what "the representation at layer l"
means. They coincide for post-LayerNorm encoders (BERT, RoBERTa, DeBERTa) and
diverge for pre-LayerNorm models (GPT-2, ModernBERT, GPT-J):

  A  "what propagates"       the raw residual stream, unnormalized.
                             Elhage et al. (2021), Ferrando et al. (2024) Eq. 5,
                             TransformerLens `resid_mid`/`resid_post`.

  B  "what the next sub-layer reads"
                             the residual stream after the LayerNorm that its
                             consumer applies. This is Hewitt & Manning's (2019)
                             `encoded_layers` when read on a post-LN encoder.

In a post-LN block, `x_mid = LN1(attn + x)` is handed straight to the feed-forward
and `x_out = LN2(ffn + x_mid)` straight to the next block's attention, so A == B
exactly. In a pre-LN block nothing normalizes the stream itself, so A and B differ
by the consumer's LayerNorm -- and each checkpoint has exactly one consumer, which
makes B unambiguous:

    GPT-2 block k:  x_mid = x + attn(ln_1(x))        -> consumed by mlp  via ln_2^(k)
                    x_out = x_mid + mlp(ln_2(x_mid)) -> consumed by attn via ln_1^(k+1)
    final block:    x_out                            -> consumed by ln_f -> unembedding

Because LayerNorm's learned affine (gamma, beta) is absorbed by the structural
probe -- beta cancels in the difference (h_i - h_j) and gamma is a diagonal map the
probe matrix can represent -- the only part of this transform the probe cannot undo
is the per-token centering and rescaling by 1/sigma(z). That is precisely the
quantity the A/B comparison isolates.

Checkpoint 0 (raw token embeddings, no position) has no consumer and is copied
through unchanged; it is not meaningful under convention B.

Usage:
  python apply_consuming_layernorm.py IN.hdf5 OUT.hdf5 gpt2
  python apply_consuming_layernorm.py IN.hdf5 OUT.hdf5 gpt2 --validate-only --conllx dev.conllx
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

# model_type -> (attribute path to the block list, has_midblock).
# Only pre-LayerNorm models need this transform; post-LN models are rejected
# because A and B already agree and silently producing a "B" file for them would
# invite double-normalization.
#
#   gpt2  serial blocks, 2+2L layout, separate ln_1 (attn) and ln_2 (mlp)
#   gptj  PARALLEL residual: h_out = h + attn(ln_1(h)) + mlp(ln_1(h)).
#         One shared ln_1 feeds both sub-layers and no mid-block state exists,
#         so the layout is 1+L and every checkpoint's consumer is an ln_1.
PRE_LN_BLOCKS = {
    'gpt2': ('h', True),
    'gptj': ('h', False),
    'modernbert': ('layers', True),
}

# Post-LayerNorm encoders. For these the LITERATURE conventions A and B coincide
# (both are the post-LayerNorm value that HF returns as hidden_states), but
# convert_raw_to_hf_all_checkpoints.py deliberately stores the PRE-LayerNorm
# accumulator instead -- a third quantity, neither A nor B. Applying the
# consuming LayerNorm to that stored accumulator recovers exactly the post-LN
# value, i.e. Hewitt & Manning's original convention.
POST_LN_BLOCKS = {
    'bert': 'encoder.layer',
    'roberta': 'encoder.layer',
    'deberta-v2': 'encoder.layer',
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
    raise TypeError(f'no tensor in hook payload of type {type(x)}')


def register_input_hook(module, store, key):
    '''Pre-hook that records a module's hidden-states input, positional or keyword.

    Sub-modules differ in how they are called -- GPT-2 passes hidden_states
    positionally to `attn`, GPT-J passes it as a keyword -- so read both rather
    than assuming a calling convention.
    '''
    def hook(m, args, kwargs):
        t = None
        if args:
            for e in args:
                if torch.is_tensor(e):
                    t = e
                    break
        if t is None:
            t = kwargs.get('hidden_states')
        if not torch.is_tensor(t):
            raise TypeError(
                f'{type(m).__name__}: no hidden-states tensor in args {type(args)} '
                f'or kwargs {sorted(kwargs)}')
        store[key] = t.detach().clone()
    return module.register_forward_pre_hook(hook, with_kwargs=True)


def consumer_norms(model, mtype, n_layers):
    '''Return, for each checkpoint index, the LayerNorm module its consumer applies
    (or None to copy the checkpoint through unchanged).

    gpt2 (2+2L layout written by convert_raw_to_hf_all_checkpoints.py):
        0        raw token embeddings          (no consumer)
        1        embedding output              -> ln_1 of block 0
        2 + 2k   post-attention of block k     -> ln_2 of block k
        3 + 2k   post-block of block k         -> ln_1 of block k+1, or ln_f at the end

    gptj (1+L layout; attention and MLP run in parallel off one shared ln_1, so
    there is no mid-block state and every consumer is an ln_1):
        0        embedding output              -> ln_1 of block 0
        k        post-block of block k-1       -> ln_1 of block k, or ln_f at the end
    '''
    if mtype in POST_LN_BLOCKS:
        # 2+2L layout over the stored PRE-LayerNorm accumulators:
        #   0        raw token embeddings                  (no consumer)
        #   1        embedding output -- ALREADY normalised by the embedding
        #            LayerNorm, and read directly by block 0's attention, so it
        #            needs no further transform
        #   2 + 2k   post-attention accumulator -> attention.output.LayerNorm of block k
        #   3 + 2k   post-block accumulator     -> output.LayerNorm of block k
        blocks = resolve(model, POST_LN_BLOCKS[mtype])
        norms = {0: None, 1: None}
        for k in range(n_layers):
            norms[2 + 2 * k] = blocks[k].attention.output.LayerNorm
            norms[3 + 2 * k] = blocks[k].output.LayerNorm
        return norms

    if mtype not in PRE_LN_BLOCKS:
        raise ValueError(f"no consumer-LayerNorm map registered for '{mtype}'")
    blocks_path, _ = PRE_LN_BLOCKS[mtype]
    blocks = resolve(model, blocks_path)

    if mtype == 'gpt2':
        norms = {0: None, 1: blocks[0].ln_1}
        for k in range(n_layers):
            norms[2 + 2 * k] = blocks[k].ln_2
            norms[3 + 2 * k] = blocks[k + 1].ln_1 if k + 1 < n_layers else model.ln_f
        return norms

    if mtype == 'modernbert':
        # Same 2+2L shape as gpt2, but the sub-layer norms are attn_norm/mlp_norm
        # and layer 0's attn_norm is nn.Identity (the embedding layer already
        # normalises), so checkpoint 1 needs no transform.
        norms = {0: None, 1: None}
        for k in range(n_layers):
            norms[2 + 2 * k] = blocks[k].mlp_norm
            norms[3 + 2 * k] = (blocks[k + 1].attn_norm if k + 1 < n_layers
                                else model.final_norm)
        return norms

    # gptj: checkpoint 0 is the embedding OUTPUT (not the raw embedding), so unlike
    # gpt2 every checkpoint here has a consumer and none is copied through.
    norms = {k: blocks[k].ln_1 for k in range(n_layers)}
    norms[n_layers] = model.ln_f
    return norms


def validate(model, tokenizer, mtype, n_layers, norms, sentences, device, tol=2e-5):
    '''Check the transform against what the sub-layers actually receive.

    Rather than trusting a reading of the block's forward(), this pre-hooks the
    consumer modules themselves (`mlp`, `attn`, `ln_f`) and compares their true
    inputs to norms[idx] applied to the convention-A checkpoint. That tests the
    definition of convention B directly.
    '''
    post_ln = mtype in POST_LN_BLOCKS
    if post_ln:
        blocks, has_mid = resolve(model, POST_LN_BLOCKS[mtype]), True
    else:
        blocks_path, has_mid = PRE_LN_BLOCKS[mtype]
        blocks = resolve(model, blocks_path)
    mid, post, blk_in, true_b = {}, {}, {}, {}
    handles = []

    # Hooks must capture CLONES, and every comparison must happen after the hooks
    # are removed: several of the norms[] modules (notably ln_f) carry hooks of
    # their own, so applying one to compute an expected value would otherwise
    # re-fire its hook and clobber the recorded value.
    for k, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(
            lambda m, i, out, k=k: post.__setitem__(k, first_tensor(out).detach().clone())))
        if post_ln:
            # A = the accumulator entering each LayerNorm; true B = what the next
            # sub-layer is handed: the feed-forward for the mid checkpoint, the
            # next block for the post checkpoint (that is just the block output).
            handles.append(register_input_hook(blk.attention.output.LayerNorm, mid, k))
            handles.append(register_input_hook(blk.output.LayerNorm, blk_in, k))
            handles.append(register_input_hook(blk.intermediate, true_b, 2 + 2 * k))
        elif has_mid:
            # gpt2 calls the mid-block norm ln_2; modernbert calls it mlp_norm.
            mid_norm = getattr(blk, 'ln_2', None) or blk.mlp_norm
            handles.append(register_input_hook(mid_norm, mid, k))
            # true convention-B values: the tensors the consumers are handed
            handles.append(register_input_hook(blk.mlp, true_b, 2 + 2 * k))
            handles.append(register_input_hook(blk.attn, true_b, 1 + 2 * k))
        else:
            handles.append(register_input_hook(blk, blk_in, k))
            # parallel residual: attn and mlp both read ln_1(x), so the attn input
            # IS the convention-B value of this block's input checkpoint.
            handles.append(register_input_hook(blk.attn, true_b, k))
    if not post_ln:
        last = (1 + 2 * n_layers) if has_mid else n_layers
        final_norm = getattr(model, 'ln_f', None)
        if final_norm is None:
            final_norm = model.final_norm          # modernbert
        handles.append(final_norm.register_forward_hook(
            lambda m, i, out: true_b.__setitem__(
                last, first_tensor(out).detach().clone())))

    snapshots = []
    try:
        for tokens in sentences:
            mid.clear(); post.clear(); blk_in.clear(); true_b.clear()
            enc = tokenizer(natural_sentence(tokens), return_tensors='pt').to(device)
            with torch.no_grad():
                model(**enc)
            a, tb = {}, {i: v[0] for i, v in true_b.items()}
            if post_ln:
                for k in range(n_layers):
                    a[2 + 2 * k] = mid[k][0]
                    a[3 + 2 * k] = blk_in[k][0]
                    tb[3 + 2 * k] = post[k][0]
            elif has_mid:
                for k in range(n_layers):
                    a[2 + 2 * k] = mid[k][0]
                    a[3 + 2 * k] = post[k][0]
            else:
                a[0] = blk_in[0][0]
                for k in range(n_layers):
                    a[k + 1] = post[k][0]
            snapshots.append((a, tb))
    finally:
        for h in handles:
            h.remove()

    worst, checked = 0.0, 0
    for a, tb in snapshots:
        for idx, ln in norms.items():
            if ln is None or idx not in tb or idx not in a:
                continue
            with torch.no_grad():
                got = ln(a[idx])
            worst = max(worst, (got - tb[idx]).abs().max().item())
            checked += 1

    print(f'validation: {checked} checkpoint comparisons over {len(snapshots)} sentences')
    print(f'validation: max |transform(A) - true consumer input| = {worst:.3e}')
    if worst > tol:
        raise RuntimeError(
            f'convention-B transform disagrees with the sub-layer inputs '
            f'(max |delta| = {worst:.3e} > {tol:.1e}). Refusing to write.')
    print('validation PASSED')


def read_conllx_sentences(path, limit=None):
    sentences, buf = [], []
    for line in open(path):
        line = line.strip()
        if line.startswith('#'):
            continue
        if not line:
            if buf:
                sentences.append(buf)
                buf = []
                if limit and len(sentences) >= limit:
                    return sentences
        else:
            buf.append(line.split('\t')[1])
    if buf:
        sentences.append(buf)
    return sentences


def norms_to_tensors(norms):
    '''Reduce the LayerNorm modules to plain (weight, bias, eps) tuples.

    Lets the transform run without instantiating the model: for a 6B-parameter
    model that is the difference between a ~24 GB job and a ~4 GB one.
    '''
    out = {}
    for idx, ln in norms.items():
        if ln is None:
            out[idx] = None
        else:
            out[idx] = (ln.weight.detach().float().cpu(),
                        None if ln.bias is None else ln.bias.detach().float().cpu(),
                        float(ln.eps))
    return out


def apply_saved_norm(spec, x):
    if spec is None:
        return x
    w, b, eps = spec
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    y = (x - mu) / torch.sqrt(var + eps) * w
    return y if b is None else y + b


def main():
    argp = ArgumentParser()
    argp.add_argument('input_path', help='convention-A hdf5')
    argp.add_argument('output_path', help='convention-B hdf5 to write')
    argp.add_argument('model_name')
    argp.add_argument('--conllx', default=None,
                      help='conllx file to draw validation sentences from')
    argp.add_argument('--validate-sentences', type=int, default=25)
    argp.add_argument('--validate-only', action='store_true')
    argp.add_argument('--revision', default=None)
    argp.add_argument('--model-dtype', default='float32', choices=['float16', 'float32'])
    argp.add_argument('--dump-norms', default=None,
                      help='write the consuming LayerNorm parameters here and exit')
    argp.add_argument('--norms', default=None,
                      help='transform using parameters from --dump-norms, without '
                           'loading the model (skips validation)')
    args = argp.parse_args()

    if args.norms:
        # Lightweight path: no model, no tokenizer, no validation.
        blob = torch.load(args.norms, weights_only=False)
        norms, n_states = blob['norms'], blob['n_states']
        print(f"using saved norms from {args.norms} "
              f"({blob['model_name']}, {n_states} checkpoints)")
        saved = True
    else:
        saved = False
        wdtype = torch.float16 if args.model_dtype == 'float16' else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.revision)
        model = AutoModel.from_pretrained(args.model_name, dtype=wdtype,
                                          low_cpu_mem_usage=True,
                                          revision=args.revision).eval()
        mtype = model.config.model_type
        if mtype in POST_LN_BLOCKS:
            n_layers, has_mid = len(resolve(model, POST_LN_BLOCKS[mtype])), True
        elif mtype in PRE_LN_BLOCKS:
            blocks_path, has_mid = PRE_LN_BLOCKS[mtype]
            n_layers = len(resolve(model, blocks_path))
        else:
            raise ValueError(
                f"no consumer-LayerNorm map registered for model_type '{mtype}'. "
                f"Add one after checking where that architecture normalises; "
                f"guessing would silently mis-scale every checkpoint.")
        norms = consumer_norms(model, mtype, n_layers)
        n_states = (2 + 2 * n_layers) if has_mid else (1 + n_layers)

        if args.conllx:
            sents = read_conllx_sentences(args.conllx, limit=args.validate_sentences)
            print(f'validating on {len(sents)} sentences from {args.conllx}')
            # Validate in the model's own dtype: the check applies the very same
            # LayerNorm module to the very same captured tensor, so it is exact
            # even in fp16 -- and avoids materialising a 6B model in fp32.
            validate(model, tokenizer, mtype, n_layers, norms, sents, 'cpu')
        else:
            print('WARNING: no --conllx given, skipping validation')

        if args.dump_norms:
            torch.save({'norms': norms_to_tensors(norms), 'n_states': n_states,
                        'model_name': args.model_name, 'model_type': mtype},
                       args.dump_norms)
            print(f'wrote consuming-LayerNorm parameters to {args.dump_norms}')
            return
        if args.validate_only:
            return
        norms = norms_to_tensors(norms)

    with h5py.File(args.input_path, 'r') as fin, h5py.File(args.output_path, 'w') as fout:
        got_states = int(fin.attrs.get('n_checkpoints', n_states))
        if got_states != n_states:
            raise ValueError(f'input has {got_states} checkpoints, expected {n_states}')
        keys = [k for k in fin.keys()]
        print(f'transforming {len(keys)} sentences, {n_states} checkpoints each')
        for n, key in enumerate(keys):
            raw = np.asarray(fin[key])
            # LayerNorm is applied in float32 regardless of storage dtype, then
            # cast back, so a float16 store does not lose precision mid-transform.
            arr = torch.tensor(raw, dtype=torch.float)
            out = torch.empty_like(arr)
            with torch.no_grad():
                for idx in range(n_states):
                    out[idx] = apply_saved_norm(norms[idx], arr[idx])
            fout.create_dataset(key, data=out.numpy().astype(raw.dtype))
            if (n + 1) % 2000 == 0:
                print(f'  {n + 1}/{len(keys)}', flush=True)

        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        fout.attrs['convention'] = 'B:consumed'
        fout.attrs['convention_note'] = (
            'residual stream after the LayerNorm its consumer applies; any '
            'checkpoint with no consumer (gpt2 idx 0, the raw token embedding) '
            'is copied through unchanged')
    print('Done.')


if __name__ == '__main__':
    main()
