"""
This module handles the reading of conllx files and hdf5 embeddings.

Specifies Dataset classes, which offer PyTorch Dataloaders for the
train/dev/test splits.
"""
import os
import re
from collections import namedtuple, defaultdict

from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import h5py
import Levenshtein as _lev_mod

# ---------------------------------------------------------------------------
# Module-level alignment utilities
# Defined here so that scripts/precompute_alignments.py can import them
# directly without duplicating code.
# ---------------------------------------------------------------------------

# PTB surface normalisation constants used by natural_sentence().
# scripts/convert_raw_to_bert_natural_sentences.py imports natural_sentence
# directly from here, so extraction and alignment always agree.
_PTB_MAP = {
    '-LRB-': '(', '-RRB-': ')', '-LSB-': '[', '-RSB-': ']', '-LCB-': '{', '-RCB-': '}',
    '``': '"', "''": '"',
    '`': "'",   # single open quote
}
_PTB_OPEN_TOKENS  = {'``', '`', '-LRB-', '-LSB-', '-LCB-'}
_PTB_CLOSE_TOKENS = {"''", "'", '-RRB-', '-RSB-', '-RCB-'}
_SURFACE_OPEN = {'(', '[', '{'}
_SURFACE_CLOSE    = {
    "'ll","'re","'ve","n't","'s","'LL","'RE","'VE","N'T","'S",
    '.', ',', ';', ':', '!', '?',')', ']','}','%'
}



def natural_sentence(tokens):
  '''Convert a PTB token list to the natural-language string used during BERT extraction.

  Replicates de_ptb_sentence() in scripts/convert_raw_to_bert_natural_sentences.py.
  '''
  parts = []
  for i, tok in enumerate(tokens):
    surface  = _PTB_MAP.get(tok, tok)
    next_tok = tokens[i + 1] if i < len(tokens) - 1 else '<EOS>'
    is_opener     = tok in _PTB_OPEN_TOKENS or tok in _SURFACE_OPEN
    is_next_closer = (next_tok in _PTB_CLOSE_TOKENS or
                      (next_tok not in _PTB_MAP and next_tok in _SURFACE_CLOSE))
    use_space = not is_opener and not is_next_closer and i < len(tokens) - 1
    parts.append(surface + (' ' if use_space else ''))
  return ''.join(parts)


def levenshtein_matrix(string1, string2):
  '''Return a (len(string1), len(string2)) binary alignment matrix from Levenshtein opcodes.'''
  mtx = torch.zeros(len(string1), len(string2))
  for opcode_type, s1b, s1e, s2b, s2e in _lev_mod.opcodes(string1, string2):
    if opcode_type in {'equal', 'replace'}:
      for k in range(s1e - s1b):
        mtx[s1b + k, s2b + k] = 1
    elif opcode_type == 'delete':
      for k in range(s1e - s1b):
        mtx[s1b + k, max(s2b, 0)] = 1
    elif opcode_type == 'insert':
      for k in range(s2e - s2b):
        mtx[max(s1b, 0), s2b + k] = 1
  return mtx


def token_to_character_alignment(tokens):
  '''Return (n_tokens, total_chars) row-normalised token-to-character matrix.'''
  total = sum(len(t) for t in tokens)
  rows, cum = [], 0
  for token in tokens:
    row = torch.zeros(total)
    for k, ch in enumerate(token):
      if ch != ' ':
        row[cum + k] = 1
    row = row / row.sum().clamp(min=1e-8)
    rows.append(row)
    cum += len(token)
  return torch.stack(rows)


_URL_RE   = re.compile(r'https?://|^www\.', re.I)
_EMAIL_RE = re.compile(r'^\S+@\S+\.\S+$')


def is_natural_sentence_ud(tokens):
  """Return True if the CoNLL-U token list looks like a natural sentence.

  Filters out content that is structurally trivial or harmful for BERT:
    - too short  : ≤ 2 tokens (no meaningful parse distances)
    - no letters : pure punctuation / separator / date lines
    - URL token  : any token starting with http(s):// or www.
    - email token: any token that is a bare email address
  """
  if len(tokens) < 3:
    return False
  if not any(re.search(r'[A-Za-z]', t) for t in tokens):
    return False
  if any(_URL_RE.search(t) for t in tokens):
    return False
  if any(_EMAIL_RE.match(t) for t in tokens):
    return False
  return True


def natural_sentence_ud(tokens, misc_fields):
  """Convert a CoNLL-U token list to the natural-language string for BERT extraction.

  Uses SpaceAfter=No from the MISC field (field 9 in CoNLL-U) for exact spacing.
  Unlike natural_sentence(), no PTB normalisation is needed — UD tokens are
  already surface forms.

  Args:
    tokens:      list of FORM strings (field 1 in CoNLL-U)
    misc_fields: list of MISC strings (field 9 in CoNLL-U); '_' means no annotation
  Returns:
    A single string — the sentence as presented to BERT.
  """
  parts = []
  for i, (tok, misc) in enumerate(zip(tokens, misc_fields)):
    add_space = (i < len(tokens) - 1) and ('SpaceAfter=No' not in misc)
    parts.append(tok + (' ' if add_space else ''))
  return ''.join(parts)


def hface_alignment_ud(tokenized_sent, untokenized_sent, misc_fields):
  """(n_sub_no_cls_sep, n_ud_words) column-normalised alignment matrix for CoNLL-U data.

  Single-step Levenshtein: subwords → UD surface string (SpaceAfter-aware).
  Simpler than hface_alignment_deptb because UD tokens are already surface form,
  so no PTB→natural intermediate mapping is needed.

  Args:
    tokenized_sent:   list of subword strings including [CLS] / [SEP]
    untokenized_sent: list of UD FORM strings
    misc_fields:      list of MISC strings (field 9) for spacing information
  Returns:
    Tensor of shape (n_sub_no_cls_sep, n_ud_words), columns sum to 1.
  """
  ud_ws = []
  for i, (tok, misc) in enumerate(zip(untokenized_sent, misc_fields)):
    space = ' ' if (i < len(untokenized_sent) - 1 and 'SpaceAfter=No' not in misc) else ''
    ud_ws.append(tok + space)
  natural_str    = ''.join(ud_ws)
  ud_tok_to_char = token_to_character_alignment(ud_ws)

  hface_tokens      = [t for t in tokenized_sent if t not in ('[CLS]', '[SEP]')]
  hface_ws          = [t + (' ' if i < len(hface_tokens) - 1 else '')
                       for i, t in enumerate(hface_tokens)]
  hface_tok_to_char = token_to_character_alignment(hface_ws)
  hface_string      = ' '.join(hface_tokens)

  lev      = levenshtein_matrix(hface_string, natural_str)
  unnorm   = hface_tok_to_char @ lev @ ud_tok_to_char.t()
  col_sums = unnorm.sum(dim=0, keepdim=True).clamp(min=1e-8)
  return unnorm / col_sums


def hface_alignment_deptb(tokenized_sent, untokenized_sent):
  '''(n_sub_no_cls_sep, n_ptb_words) column-normalised alignment matrix.

  Two-step Levenshtein: subwords → natural string (what BERT saw) → PTB string.
  Used for HDF5 files extracted by scripts/convert_raw_to_bert_natural_sentences.py.
  '''
  natural_str = natural_sentence(list(untokenized_sent))

  ptb_ws          = [t + (' ' if i < len(untokenized_sent) - 1 else '')
                     for i, t in enumerate(untokenized_sent)]
  ptb_raw_string  = ''.join(ptb_ws)
  ptb_tok_to_char = token_to_character_alignment(ptb_ws)

  hface_tokens    = [t for t in tokenized_sent if t not in ('[CLS]', '[SEP]')]
  hface_ws        = [t + (' ' if i < len(hface_tokens) - 1 else '')
                     for i, t in enumerate(hface_tokens)]
  hface_tok_to_char = token_to_character_alignment(hface_ws)
  hface_string    = ' '.join(hface_tokens)

  lev1   = levenshtein_matrix(hface_string,  natural_str)
  lev2   = levenshtein_matrix(natural_str,   ptb_raw_string)
  unnorm = hface_tok_to_char @ lev1 @ lev2 @ ptb_tok_to_char.t()
  col_sums = unnorm.sum(dim=0, keepdim=True).clamp(min=1e-8)
  return unnorm / col_sums


def hface_alignment_natural(natural_str, offsets, untokenized_sent):
  '''(n_sub_no_specials, n_ptb_words) column-normalised alignment, tokenizer-agnostic.

  Same two-stage idea as hface_alignment_deptb -- de-PTBify to the natural string
  the model actually saw, then map back to PTB tokens -- but step 1 uses the fast
  tokenizer's exact character offsets instead of a fuzzy Levenshtein over raw token
  strings. That makes it independent of the subword marker convention (WordPiece
  '##', BPE 'Ġ', SentencePiece '▁') and of the special-token names ([CLS]/[SEP],
  <s>/</s>, ...), both of which hface_alignment_deptb hardcodes for BERT.

  Step 2 (natural string -> PTB string) remains a character Levenshtein, because
  de-PTBification genuinely rewrites characters ('-LRB-' -> '(') and closes spaces.

  Equivalence: on 400 PTB dev sentences this reproduces hface_alignment_deptb for
  BERT-base to within 1e-6 (identical matrices), so switching does not perturb
  previously computed BERT results.

  Args:
    natural_str:      the string fed to the model, from natural_sentence().
    offsets:          list of (start_char, end_char) for the non-special subwords,
                      in order, as returned by a fast tokenizer's offset_mapping.
    untokenized_sent: list of PTB token strings.
  Returns:
    Tensor of shape (len(offsets), len(untokenized_sent)); columns sum to 1.
  '''
  ptb_ws = [t + (' ' if i < len(untokenized_sent) - 1 else '')
            for i, t in enumerate(untokenized_sent)]
  ptb_tok_to_char = token_to_character_alignment(ptb_ws)
  lev2 = levenshtein_matrix(natural_str, ''.join(ptb_ws))

  sub_to_char = torch.zeros(len(offsets), len(natural_str))
  for i, (start, end) in enumerate(offsets):
    if end > start:
      sub_to_char[i, start:end] = 1.0 / (end - start)

  unnorm = sub_to_char @ lev2 @ ptb_tok_to_char.t()
  return unnorm / unnorm.sum(dim=0, keepdim=True).clamp(min=1e-8)


def resolve_hf_model_name(args):
  '''Return the HuggingFace model id to build a tokenizer from.

  Explicit configuration only. The previous behaviour guessed from
  args['model']['hidden_dim'] (768 -> bert-base-cased, 1024 -> bert-large-cased),
  which silently returns the wrong tokenizer for any other 768/1024-dim encoder
  -- deberta-v3-base, ModernBERT-base and roberta-base are all 768.
  '''
  name = args['model'].get('hf_model_name')
  if not name:
    raise ValueError(
        "config is missing model.hf_model_name; set it to the HuggingFace id whose "
        "tokenizer produced the stored embeddings (e.g. 'bert-base-cased', "
        "'roberta-base', 'microsoft/deberta-v3-base'). It is no longer inferred "
        "from hidden_dim, which cannot distinguish same-width models.")
  return name


class SimpleDataset:
  """Reads conllx files to provide PyTorch Dataloaders

  Reads the data from conllx files into namedtuple form to keep annotation
  information, and provides PyTorch dataloaders and padding/batch collation
  to provide access to train, dev, and test splits.

  Attributes:
    args: the global yaml-derived experiment config dictionary
  """
  def __init__(self, args, task, vocab={}):
    self.args = args
    self.batch_size = args['dataset']['batch_size']
    self.use_disk_embeddings = args['model']['use_disk']
    self.vocab = vocab
    self.observation_class = self.get_observation_class(self.args['dataset']['observation_fieldnames'])
    self.train_obs, self.dev_obs, self.test_obs = self.read_from_disk()
    self.train_dataset = ObservationIterator(self.train_obs, task)
    self.dev_dataset = ObservationIterator(self.dev_obs, task)
    self.test_dataset = ObservationIterator(self.test_obs, task)

  def read_from_disk(self):
    '''Reads observations from conllx-formatted files
    
    as specified by the yaml arguments dictionary and 
    optionally adds pre-constructed embeddings for them.

    Returns:
      A 3-tuple: (train, dev, test) where each element in the
      tuple is a list of Observations for that split of the dataset. 
    '''
    train_corpus_path = os.path.join(self.args['dataset']['corpus']['root'],
        self.args['dataset']['corpus']['train_path'])
    dev_corpus_path = os.path.join(self.args['dataset']['corpus']['root'],
        self.args['dataset']['corpus']['dev_path'])
    test_corpus_path = os.path.join(self.args['dataset']['corpus']['root'],
        self.args['dataset']['corpus']['test_path'])
    train_observations = self.load_conll_dataset(train_corpus_path)
    dev_observations = self.load_conll_dataset(dev_corpus_path)
    test_observations = self.load_conll_dataset(test_corpus_path)

    train_embeddings_path = os.path.join(self.args['dataset']['embeddings']['root'],
        self.args['dataset']['embeddings']['train_path'])
    dev_embeddings_path = os.path.join(self.args['dataset']['embeddings']['root'],
        self.args['dataset']['embeddings']['dev_path'])
    test_embeddings_path = os.path.join(self.args['dataset']['embeddings']['root'],
        self.args['dataset']['embeddings']['test_path'])

    if self.args['model'].get('alignment') == 'pre-aligned':
      align_root = self.args['model']['alignment_root']
      train_align_path = os.path.join(align_root, self.args['model']['alignment_train_path'])
      dev_align_path   = os.path.join(align_root, self.args['model']['alignment_dev_path'])
      test_align_path  = os.path.join(align_root, self.args['model']['alignment_test_path'])
    else:
      train_align_path = dev_align_path = test_align_path = None

    train_observations = self.optionally_add_embeddings(train_observations, train_embeddings_path, train_align_path)
    dev_observations   = self.optionally_add_embeddings(dev_observations,   dev_embeddings_path,   dev_align_path)
    test_observations  = self.optionally_add_embeddings(test_observations,  test_embeddings_path,  test_align_path)
    return train_observations, dev_observations, test_observations

  def get_observation_class(self, fieldnames):
    '''Returns a namedtuple class for a single observation.

    The namedtuple class is constructed to hold all language and annotation
    information for a single sentence or document.

    Args:
      fieldnames: a list of strings corresponding to the information in each
        row of the conllx file being read in. (The file should not have
        explicit column headers though.)
    Returns:
      A namedtuple class; each observation in the dataset will be an instance
      of this class.
    '''
    return namedtuple('Observation', fieldnames)

  def generate_lines_for_sent(self, lines):
    '''Yields batches of lines describing a sentence in conllx.

    Args:
      lines: Each line of a conllx file.
    Yields:
      a list of lines describing a single sentence in conllx.
    '''
    buf = []
    for line in lines:
      if line.startswith('#'):
        continue
      if re.match(r'^\d+[-\.]\d+', line):
        continue
      if not line.strip():
        if buf:
          yield buf
          buf = []
        else:
          continue
      else:
        buf.append(line.strip())
    if buf:
      yield buf

  def load_conll_dataset(self, filepath):
    '''Reads in a conllx file; generates Observation objects
    
    For each sentence in a conllx file, generates a single Observation
    object.

    Args:
      filepath: the filesystem path to the conll dataset
  
    Returns:
      A list of Observations 
    '''
    observations = []
    lines = (x for x in open(filepath))
    for buf in self.generate_lines_for_sent(lines):
      conllx_lines = []
      for line in buf:
        conllx_lines.append(line.strip().split('\t'))
      embeddings = [None for x in range(len(conllx_lines))]
      observation = self.observation_class(*zip(*conllx_lines), embeddings)
      observations.append(observation)
    return observations

  def add_embeddings_to_observations(self, observations, embeddings):
    '''Adds pre-computed embeddings to Observations.

    Args:
      observations: A list of Observation objects composing a dataset.
      embeddings: A list of pre-computed embeddings in the same order.

    Returns:
      A list of Observations with pre-computed embedding fields.
    '''
    embedded_observations = []
    for observation, embedding in zip(observations, embeddings):
      embedded_observation = self.observation_class(*(observation[:-1]), embedding)
      embedded_observations.append(embedded_observation)
    return embedded_observations

  def generate_token_embeddings_from_hdf5(self, args, observations, filepath, layer_index):
    '''Reads pre-computed embeddings from ELMo-like hdf5-formatted file.

    Sentences should be given integer keys corresponding to their order
    in the original file.
    Embeddings should be of the form (layer_count, sent_length, feature_count)

    Args:
      args: the global yaml-derived experiment config dictionary.
      observations: A list of Observations composing a dataset.
      filepath: The filepath of a hdf5 file containing embeddings.
      layer_index: The index corresponding to the layer of representation
          to be used. (e.g., 0, 1, 2 for ELMo0, ELMo1, ELMo2.)
    
    Returns:
      A list of numpy matrices; one for each observation.

    Raises:
      AssertionError: sent_length of embedding was not the length of the
        corresponding sentence in the dataset.
    '''
    hf = h5py.File(filepath, 'r') 
    indices = filter(lambda x: x != 'sentence_to_index', list(hf.keys()))
    single_layer_features_list = []
    for index in sorted([int(x) for x in indices]):
      observation = observations[index]
      feature_stack = hf[str(index)]
      single_layer_features = feature_stack[layer_index]
      assert single_layer_features.shape[0] == len(observation.sentence)
      single_layer_features_list.append(single_layer_features)
    return single_layer_features_list

  def integerize_observations(self, observations):
    '''Replaces strings in an Observation with integer Ids.
    
    The .sentence field of the Observation will have its strings
    replaced with integer Ids from self.vocab. 

    Args:
      observations: A list of Observations describing a dataset

    Returns:
      A list of observations with integer-lists for sentence fields
    '''
    new_observations = []
    if self.vocab == {}:
      raise ValueError("Cannot replace words with integer ids with an empty vocabulary "
          "(and the vocabulary is in fact empty")
    for observation in observations:
      sentence = tuple([self.vocab[sym] for sym in observation.sentence])
      new_observations.append(self.observation_class(sentence, *observation[1:]))
    return new_observations

  def get_train_dataloader(self, shuffle=True, use_embeddings=True):
    """Returns a PyTorch dataloader over the training dataset.

    Args:
      shuffle: shuffle the order of the dataset.
      use_embeddings: ignored

    Returns:
      torch.DataLoader generating the training dataset (possibly shuffled)
    """
    return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=shuffle)

  def get_dev_dataloader(self, use_embeddings=True):
    """Returns a PyTorch dataloader over the development dataset.

    Args:
      use_embeddings: ignored

    Returns:
      torch.DataLoader generating the development dataset
    """
    return DataLoader(self.dev_dataset, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=False)

  def get_test_dataloader(self, use_embeddings=True):
    """Returns a PyTorch dataloader over the test dataset.

    Args:
      use_embeddings: ignored

    Returns:
      torch.DataLoader generating the test dataset
    """
    return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=False)

  def optionally_add_embeddings(self, observations, pretrained_embeddings_path, alignment_path=None):
    """Does not add embeddings; see subclasses for implementations."""
    return observations

  def custom_pad(self, batch_observations):
    '''Pads sequences with 0 and labels with -1; used as collate_fn of DataLoader.
    
    Loss functions will ignore -1 labels.
    If labels are 1D, pads to the maximum sequence length.
    If labels are 2D, pads all to (maxlen,maxlen).

    Args:
      batch_observations: A list of observations composing a batch
    
    Return:
      A tuple of:
          input batch, padded
          label batch, padded
          lengths-of-inputs batch, padded
          Observation batch (not padded)
    '''
    if self.use_disk_embeddings:
      seqs = [torch.tensor(x[0].embeddings, device=self.args['device']) for x in batch_observations]
    else:
      seqs = [torch.tensor(x[0].sentence, device=self.args['device']) for x in batch_observations]
    lengths = torch.tensor([len(x) for x in seqs], device=self.args['device'])
    seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    label_shape = batch_observations[0][1].shape
    maxlen = int(max(lengths))
    label_maxshape = [maxlen for x in label_shape]
    labels = [-torch.ones(*label_maxshape, device=self.args['device']) for x in seqs]
    for index, x in enumerate(batch_observations):
      length = x[1].shape[0]
      if len(label_shape) == 1:
        labels[index][:length] = x[1]
      elif len(label_shape) == 2:
        labels[index][:length,:length] = x[1]
      else:
        raise ValueError("Labels must be either 1D or 2D right now; got either 0D or >3D")
    labels = torch.stack(labels)
    return seqs, labels, lengths, batch_observations

class ELMoDataset(SimpleDataset):
  """Dataloader for conllx files and pre-computed ELMo embeddings.

  See SimpleDataset.
  Assumes embeddings are aligned with tokens in conllx file.
  Attributes:
    args: the global yaml-derived experiment config dictionary
  """

  def optionally_add_embeddings(self, observations, pretrained_embeddings_path, alignment_path=None):
    """Adds pre-computed ELMo embeddings from disk to Observations."""
    layer_index = self.args['model']['model_layer']
    print('Loading ELMo Pretrained Embeddings from {}; using layer {}'.format(pretrained_embeddings_path, layer_index))
    embeddings = self.generate_token_embeddings_from_hdf5(self.args, observations, pretrained_embeddings_path, layer_index)
    observations = self.add_embeddings_to_observations(observations, embeddings)
    return observations

class SubwordDataset(SimpleDataset):
  """Dataloader for conllx files and pre-computed ELMo embeddings.

  See SimpleDataset.
  Assumes we have access to the subword tokenizer.
  """

  @staticmethod
  def match_tokenized_to_untokenized(tokenized_sent, untokenized_sent, connection_character='##', special_tokens=True):
    '''Aligns tokenized and untokenized sentence given subwords "##" prefixed

    Assuming that each subword token that does not start a new word is prefixed
    by two hashes, "##", computes an alignment between the un-subword-tokenized
    and subword-tokenized sentences.

    Args:
      tokenized_sent: a list of strings describing a subword-tokenized sentence
      untokenized_sent: a list of strings describing a sentence, no subword tok.
    Returns:
      A dictionary of type {int: list(int)} mapping each untokenized sentence
      index to a list of subword-tokenized sentence indices
    '''
    mapping = defaultdict(list)
    untokenized_sent_index = 0
    tokenized_sent_index = 1 if special_tokens else 0
    while (untokenized_sent_index < len(untokenized_sent) and
        tokenized_sent_index < len(tokenized_sent)):
      while (tokenized_sent_index + 1 < len(tokenized_sent) and
          tokenized_sent[tokenized_sent_index + 1].startswith(connection_character)):
        mapping[untokenized_sent_index].append(tokenized_sent_index)
        tokenized_sent_index += 1
      mapping[untokenized_sent_index].append(tokenized_sent_index)
      untokenized_sent_index += 1
      tokenized_sent_index += 1
    return mapping

  def generate_subword_embeddings_from_hdf5(self, observations, filepath, elmo_layer, subword_tokenizer=None):
    raise NotImplementedError("Instead of making a SubwordDataset, make one of the implementing classes")

class BERTDataset(SubwordDataset):
  """Dataloader for conllx files and pre-computed BERT embeddings.

  See SimpleDataset.
  Attributes:
    args: the global yaml-derived experiment config dictionary
  """

  def generate_subword_embeddings_from_hdf5(self, observations, filepath, elmo_layer, subword_tokenizer=None, alignment_path=None):
    '''Reads pre-computed subword embeddings from hdf5-formatted file.

    Sentences should be given integer keys corresponding to their order
    in the original file.
    Embeddings should be of the form (layer_count, subword_sent_length, feature_count)
    subword_sent_length is the length of the sequence of subword tokens
    when the subword tokenizer was given each canonical token (as given
    by the conllx file) independently and tokenized each. Thus, there
    is a single alignment between the subword-tokenized sentence
    and the conllx tokens.

    Args:
      args: the global yaml-derived experiment config dictionary.
      observations: A list of Observations composing a dataset.
      filepath: The filepath of a hdf5 file containing embeddings.
      layer_index: The index corresponding to the layer of representation
          to be used. (e.g., 0, 1, 2 for BERT0, BERT1, BERT2.)
      subword_tokenizer: (optional) a tokenizer used to map from
          conllx tokens to subword tokens.
    
    Returns:
      A list of numpy matrices; one for each observation.

    Raises:
      AssertionError: sent_length of embedding was not the length of the
        corresponding sentence in the dataset.
      Exit: importing pytorch_pretrained_bert has failed, possibly due 
          to downloading of prespecifed tokenizer problem. Not recoverable;
          exits immediately.
    '''
    alignment_method = self.args['model'].get('alignment')

    if alignment_method == 'pre-aligned':
      # Alignment was pre-computed by scripts/precompute_alignments.py.
      # Each sentence key holds align_mat of shape (n_sub-2, n_words); no
      # tokeniser or Levenshtein work needed here — just a matmul per sentence.
      print(f'Loading pre-computed alignments from {alignment_path}')
      hf        = h5py.File(filepath,       'r')
      align_hf  = h5py.File(alignment_path, 'r')
      indices   = list(hf.keys())
      # How many special-token rows to strip from each stored sentence. Written by
      # scripts/precompute_alignments_hf.py; defaults to the BERT-era convention of
      # one leading [CLS] and one trailing [SEP] for alignment files predating it.
      # GPT-style tokenizers add none, where stripping would delete real tokens.
      n_pre = int(align_hf.attrs.get('n_prefix_special', 1))
      n_suf = int(align_hf.attrs.get('n_suffix_special', 1))
      print(f'Stripping {n_pre} leading and {n_suf} trailing special-token rows')
      single_layer_features_list = []
      for index in tqdm(sorted([int(x) for x in indices]), desc='[loading embeddings]'):
        observation          = observations[index]
        single_layer_features = np.array(hf[str(index)][elmo_layer])   # (n_sub, hidden)
        align_mat = torch.tensor(np.array(align_hf[str(index)]), dtype=torch.float)  # (n_sub_no_specials, n_words)
        end = single_layer_features.shape[0] - n_suf
        sub_feats = torch.tensor(single_layer_features[n_pre:end],     dtype=torch.float)
        single_layer_features = align_mat.t() @ sub_feats               # (n_words, hidden)
        assert single_layer_features.shape[0] == len(observation.sentence)
        single_layer_features_list.append(single_layer_features)
      hf.close()
      align_hf.close()
      return single_layer_features_list

    if subword_tokenizer == None:
      from transformers import AutoTokenizer
      model_name = resolve_hf_model_name(self.args)
      subword_tokenizer = AutoTokenizer.from_pretrained(model_name)
      print(f'Using {model_name} tokenizer to align embeddings with PTB tokens')
    hf = h5py.File(filepath, 'r')
    indices = list(hf.keys())
    single_layer_features_list = []
    for index in tqdm(sorted([int(x) for x in indices]), desc='[aligning embeddings]'):
      observation = observations[index]
      feature_stack = hf[str(index)]
      single_layer_features = feature_stack[elmo_layer]
      untokenized_sent = observation.sentence

      if alignment_method == 'hface-deptb':
        # HDF5 was extracted on de-PTBified text; re-tokenise the same string
        natural = self.natural_sentence(list(untokenized_sent))
        tokenized_sent = ['[CLS]'] + subword_tokenizer.tokenize(natural) + ['[SEP]']
        assert single_layer_features.shape[0] == len(tokenized_sent)
        align_mat = self.hface_alignment_deptb(tokenized_sent, list(untokenized_sent))
        sub_feats = torch.tensor(np.array(single_layer_features[1:-1]), dtype=torch.float)
        single_layer_features = align_mat.t() @ sub_feats
      else:
        tokenized_sent = ['[CLS]'] + subword_tokenizer.tokenize(' '.join(untokenized_sent)) + ['[SEP]']
        assert single_layer_features.shape[0] == len(tokenized_sent)
        if alignment_method == 'hface':
          align_mat = self.hface_alignment(tokenized_sent, list(untokenized_sent))
          sub_feats = torch.tensor(np.array(single_layer_features[1:-1]), dtype=torch.float)
          single_layer_features = align_mat.t() @ sub_feats
        else:
          untok_tok_mapping = self.match_tokenized_to_untokenized(tokenized_sent, untokenized_sent)
          single_layer_features = torch.tensor([np.mean(single_layer_features[untok_tok_mapping[i][0]:untok_tok_mapping[i][-1]+1,:], axis=0) for i in range(len(untokenized_sent))])

      assert single_layer_features.shape[0] == len(observation.sentence)
      single_layer_features_list.append(single_layer_features)
    return single_layer_features_list

  def optionally_add_embeddings(self, observations, pretrained_embeddings_path, alignment_path=None):
    """Adds pre-computed BERT embeddings from disk to Observations."""
    layer_index = self.args['model']['model_layer']
    print('Loading BERT Pretrained Embeddings from {}; using layer {}'.format(pretrained_embeddings_path, layer_index))
    embeddings = self.generate_subword_embeddings_from_hdf5(observations, pretrained_embeddings_path, layer_index, alignment_path=alignment_path)
    observations = self.add_embeddings_to_observations(observations, embeddings)
    return observations

  def levenshtein_matrix(self, string1, string2):
    return levenshtein_matrix(string1, string2)

  def token_to_character_alignment(self, tokens):
    return token_to_character_alignment(tokens)

  def de_ptb_tokenize(self, tokens):
    new_tokens_with_spaces = []
    ptb_sentence_length = sum((len(tok) for tok in tokens))
    token_alignments = []
    cumulative = 0
    for i, _ in enumerate(tokens):
      token = tokens[i]
      next_token = tokens[i+1] if i < len(tokens)-1 else '<EOS>'
      if token.strip() in {"``", "''"}:
        new_token = '"'
      elif token.strip() == '-LRB-':
        new_token = '('
      elif token.strip() == '-RRB-':
        new_token = ')'
      elif token.strip() == '-LSB-':
        new_token = '['
      elif token.strip() == '-RSB-':
        new_token = ']'
      elif token.strip() == '-LCB-':
        new_token = '{'
      elif token.strip() == '-RCB-':
        new_token = '}'
      else:
        new_token = token
      # Only resolve closing brackets for the next-token spacing check; leave
      # quotes as original PTB tokens so "``" is not confused with "''" / '"'.
      _CLOSE_ONLY = {'-RRB-': ')', '-RSB-': ']', '-RCB-': '}'}
      next_surface = _CLOSE_ONLY.get(next_token.strip(), next_token.strip())
      use_space = (new_token.strip() not in {'(', '[', '{', '"', "'", '``', "''"} and
                   next_surface not in {"'ll", "'re", "'ve", "n't",
                                        "'s", "'LL", "'RE", "'VE",
                                        "N'T", "'S", '"', "'", '``', "''",
                                        ')', '}', ']', '.', ';', ':', '!', '?'}
                   and i != len(tokens) - 1)
      new_token = new_token.strip() + (' ' if use_space else '')
      new_tokens_with_spaces.append(new_token)
      new_alignment = torch.zeros(ptb_sentence_length)
      for index, char in enumerate(token):
        new_alignment[index+cumulative] = 1
      for new_char in new_token:
        token_alignments.append(new_alignment)
      cumulative += len(token)
    return new_tokens_with_spaces, torch.stack(token_alignments)

  def hface_alignment(self, tokenized_sent, untokenized_sent):
    '''Compute (n_subwords, n_words) column-normalised alignment via character-level Levenshtein.

    Improves on the ##-prefix heuristic for PTB tokens like -LRB- that split into
    subwords without ## continuations. Works with existing HDF5 (PTB-string tokenisation).
    '''
    tokens_with_spaces = [t + (' ' if i < len(untokenized_sent) - 1 else '')
                          for i, t in enumerate(untokenized_sent)]
    raw_string = ''.join(tokens_with_spaces)
    ptb_tok_to_char = token_to_character_alignment(tokens_with_spaces)

    hface_tokens = [t for t in tokenized_sent if t not in ('[CLS]', '[SEP]')]
    hface_tokens_with_spaces = [t + (' ' if i < len(hface_tokens) - 1 else '')
                                 for i, t in enumerate(hface_tokens)]
    hface_tok_to_char = token_to_character_alignment(hface_tokens_with_spaces)
    hface_string = ' '.join(hface_tokens)

    lev = levenshtein_matrix(hface_string, raw_string)
    unnorm = hface_tok_to_char @ lev @ ptb_tok_to_char.t()
    col_sums = unnorm.sum(dim=0, keepdim=True).clamp(min=1e-8)
    return unnorm / col_sums  # (n_sub, n_ptb), columns sum to 1

  def natural_sentence(self, tokens):
    return natural_sentence(tokens)

  def natural_sentence_ud(self, tokens, misc_fields):
    return natural_sentence_ud(tokens, misc_fields)

  def hface_alignment_deptb(self, tokenized_sent, untokenized_sent):
    return hface_alignment_deptb(tokenized_sent, untokenized_sent)

  def hface_alignment_ud(self, tokenized_sent, untokenized_sent, misc_fields):
    return hface_alignment_ud(tokenized_sent, untokenized_sent, misc_fields)



class GPTDataset(SubwordDataset):
  """Dataloader for conllx files and pre-computed BERT embeddings.

  See SimpleDataset.
  Attributes:
    args: the global yaml-derived experiment config dictionary
  """
  @staticmethod
  def convert_offsets_to_mapping(offsets, words):
    # convert the character offsets to word offsets
    char_to_word = []
    word_idx = 0
    sentence = ' '.join(words)
    for char_idx, c in enumerate(sentence):
      if c == ' ':
        word_idx += 1
      char_to_word.append(word_idx)
    # convert token idx to word idx 
    untok_to_tok_mapping = [[] for _ in range(len(words))]
    for i, (start, end) in enumerate(offsets):
      untok_to_tok_mapping[char_to_word[start]].append(i)
    return untok_to_tok_mapping

  def generate_subword_embeddings_from_hdf5(self, observations, filepath, elmo_layer, subword_tokenizer=None):
    '''Reads pre-computed subword embeddings from hdf5-formatted file.

    Sentences should be given integer keys corresponding to their order
    in the original file.
    Embeddings should be of the form (layer_count, subword_sent_length, feature_count)
    subword_sent_length is the length of the sequence of subword tokens
    when the subword tokenizer was given each canonical token (as given
    by the conllx file) independently and tokenized each. Thus, there
    is a single alignment between the subword-tokenized sentence
    and the conllx tokens.

    Args:
      args: the global yaml-derived experiment config dictionary.
      observations: A list of Observations composing a dataset.
      filepath: The filepath of a hdf5 file containing embeddings.
      layer_index: The index corresponding to the layer of representation
          to be used. (e.g., 0, 1, 2 for BERT0, BERT1, BERT2.)
      bpe_tokenizer: (optional) a tokenizer used to map from
          conllx tokens to subword tokens.
    
    Returns:
      A list of numpy matrices; one for each observation.

    Raises:
      AssertionError: sent_length of embedding was not the length of the
        corresponding sentence in the dataset.
      Exit: importing pytorch_pretrained_bert has failed, possibly due 
          to downloading of prespecifed tokenizer problem. Not recoverable;
          exits immediately.
    '''
    if bpe_tokenizer == None:
      from transformers import GPT2TokenizerFast
      model_size = 'gpt2'  
      bpe_tokenizer = GPT2TokenizerFast.from_pretrained(model_size)
      print(f'Using {model_size} tokenizer to align embeddings with PTB tokens')
    
    hf = h5py.File(filepath, 'r')
    indices = list(hf.keys())
    single_layer_features_list = []
    for index in tqdm(sorted([int(x) for x in indices]), desc='[aligning embeddings]'):
      observation = observations[index]
      feature_stack = hf[str(index)]
      single_layer_features = feature_stack[elmo_layer]
      tokenized_sent = bpe_tokenizer.tokenize(' '.join(observation.sentence))
      untokenized_sent = observation.sentence
      enc = bpe_tokenizer(' '.join(observation.sentence), return_tensors='pt', return_offsets_mapping=True)
      untok_tok_mapping = self.convert_offsets_to_mapping(enc['offset_mapping'][0], observation.sentence)
      assert single_layer_features.shape[0] == len(tokenized_sent)
      single_layer_features = torch.tensor([np.mean(single_layer_features[untok_tok_mapping[i][0]:untok_tok_mapping[i][-1]+1,:], axis=0) for i in range(len(untokenized_sent))])
      assert single_layer_features.shape[0] == len(observation.sentence)
      single_layer_features_list.append(single_layer_features)
    return single_layer_features_list

  def optionally_add_embeddings(self, observations, pretrained_embeddings_path, alignment_path=None):
    """Adds pre-computed BERT embeddings from disk to Observations."""
    layer_index = self.args['model']['model_layer']
    print(f'Loading GPT-2 Pretrained Embeddings from {pretrained_embeddings_path}; using layer {layer_index}')    
    embeddings = self.generate_subword_embeddings_from_hdf5(observations, pretrained_embeddings_path, layer_index)
    observations = self.add_embeddings_to_observations(observations, embeddings)
    return observations


class RobertaDataset(SubwordDataset):
  """Dataloader for conllx files and pre-computed BERT embeddings.

  See SimpleDataset.
  Attributes:
    args: the global yaml-derived experiment config dictionary
  """
  @staticmethod
  def convert_offsets_to_mapping(offsets, words):
    # convert the character offsets to word offsets
    char_to_word = []
    word_idx = 0
    sentence = ' '.join(words)
    for char_idx, c in enumerate(sentence):
      if c == ' ':
        word_idx += 1
      char_to_word.append(word_idx)
    # convert token idx to word idx 
    untok_to_tok_mapping = [[] for _ in range(len(words))]
    for i, (start, end) in enumerate(offsets):
      untok_to_tok_mapping[char_to_word[start]].append(i)
    return untok_to_tok_mapping

  def generate_subword_embeddings_from_hdf5(self, observations, filepath, elmo_layer, bpe_tokenizer=None):
    '''Reads pre-computed subword embeddings from hdf5-formatted file.

    Sentences should be given integer keys corresponding to their order
    in the original file.
    Embeddings should be of the form (layer_count, subword_sent_length, feature_count)
    subword_sent_length is the length of the sequence of subword tokens
    when the subword tokenizer was given each canonical token (as given
    by the conllx file) independently and tokenized each. Thus, there
    is a single alignment between the subword-tokenized sentence
    and the conllx tokens.

    Args:
      args: the global yaml-derived experiment config dictionary.
      observations: A list of Observations composing a dataset.
      filepath: The filepath of a hdf5 file containing embeddings.
      layer_index: The index corresponding to the layer of representation
          to be used. (e.g., 0, 1, 2 for BERT0, BERT1, BERT2.)
      bpe_tokenizer: (optional) a tokenizer used to map from
          conllx tokens to subword tokens.
    
    Returns:
      A list of numpy matrices; one for each observation.

    Raises:
      AssertionError: sent_length of embedding was not the length of the
        corresponding sentence in the dataset.
      Exit: importing pytorch_pretrained_bert has failed, possibly due 
          to downloading of prespecifed tokenizer problem. Not recoverable;
          exits immediately.
    '''
    if bpe_tokenizer == None:
      from transformers import AutoTokenizer
      model_name = resolve_hf_model_name(self.args)
      bpe_tokenizer = AutoTokenizer.from_pretrained(model_name)
      print(f'Using {model_name} tokenizer to align embeddings with PTB tokens')

    hf = h5py.File(filepath, 'r')
    indices = list(hf.keys())
    single_layer_features_list = []
    for index in tqdm(sorted([int(x) for x in indices]), desc='[aligning embeddings]'):
      observation = observations[index]
      feature_stack = hf[str(index)]
      single_layer_features = feature_stack[elmo_layer]
      tokenized_sent = bpe_tokenizer.tokenize(' '.join(observation.sentence))
      untokenized_sent = observation.sentence
      enc = bpe_tokenizer(' '.join(observation.sentence), return_tensors='pt', return_offsets_mapping=True)
      untok_tok_mapping = self.convert_offsets_to_mapping(enc['offset_mapping'][0][1:-1], observation.sentence)
      assert single_layer_features.shape[0] == len(tokenized_sent)
      single_layer_features = torch.tensor([np.mean(single_layer_features[untok_tok_mapping[i][0]:untok_tok_mapping[i][-1]+1,:], axis=0) for i in range(len(untokenized_sent))])
      assert single_layer_features.shape[0] == len(observation.sentence)
      single_layer_features_list.append(single_layer_features)
    return single_layer_features_list

  def optionally_add_embeddings(self, observations, pretrained_embeddings_path, alignment_path=None):
    """Adds pre-computed BERT embeddings from disk to Observations."""
    layer_index = self.args['model']['model_layer']
    print(f'Loading GPT-2 Pretrained Embeddings from {pretrained_embeddings_path}; using layer {layer_index}')    
    embeddings = self.generate_subword_embeddings_from_hdf5(observations, pretrained_embeddings_path, layer_index)
    observations = self.add_embeddings_to_observations(observations, embeddings)
    return observations


class ObservationIterator(Dataset):
  """ List Container for lists of Observations and labels for them.

  Used as the iterator for a PyTorch dataloader.
  """

  def __init__(self, observations, task):
    self.observations = observations
    self.set_labels(observations, task)

  def set_labels(self, observations, task):
    """ Constructs aand stores label for each observation.

    Args:
      observations: A list of observations describing a dataset
      task: a Task object which takes Observations and constructs labels.
    """
    self.labels = []
    for observation in tqdm(observations, desc='[computing labels]'):
      self.labels.append(task.labels(observation))

  def __len__(self):
    return len(self.observations)

  def __getitem__(self, idx):
    return self.observations[idx], self.labels[idx]

