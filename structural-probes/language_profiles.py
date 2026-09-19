"""Language-specific constants, selected by name ('english' is the default everywhere).

The English values are the literals the pipeline has always used; they were moved
here unchanged, so every run that does not name a language behaves exactly as
before. English text for the model is still built by data.natural_sentence,
which is not touched; Chinese text is built by join_cjk below.

A run selects its language with `dataset: {language: chinese}` in the probe
config and `--language chinese` on the extraction and alignment scripts.
"""
import re

# Part-of-speech tags whose words are kept out of the minimum spanning tree and
# of the root prediction, i.e. out of UUAS and UASL. English: the Penn Treebank
# punctuation tags, as hardcoded in Hewitt & Manning's reporter. Chinese: the
# Chinese Treebank tags every punctuation mark PU.
PUNCT_POS = {
    'english': frozenset(["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]),
    'chinese': frozenset(['PU']),
}

# Tag prefixes that mark a verbal head (the verb-head-share control).
VERB_TAG_PREFIXES = {
    'english': ('VB',),
    'chinese': ('VV', 'VA', 'VC', 'VE'),
}

LANGUAGES = tuple(PUNCT_POS)


def language_of(config):
    """The language a probe config names; 'english' when it names none."""
    return (config.get('dataset') or {}).get('language', 'english')


def punct_pos(language='english'):
    try:
        return PUNCT_POS[language]
    except KeyError:
        raise ValueError(f'unknown language {language!r}; known: {LANGUAGES}')


_ASCII_ALNUM = re.compile(r'[A-Za-z0-9]')


def join_cjk_spans(tokens):
    """Chinese Treebank words -> (the text the model reads, each word's char span).

    Words are concatenated without a separator, as Chinese is written, except that
    a single space separates two adjacent words when the first ends and the second
    begins with an ASCII letter or digit (Latin names, numbers written in ASCII),
    which would otherwise fuse into one string. Spans are half-open [start, end)
    indices into the returned text, so the word of every character is known
    exactly and no string alignment is needed.
    """
    parts, spans, pos = [], [], 0
    for i, tok in enumerate(tokens):
        if i and _ASCII_ALNUM.match(tokens[i - 1][-1]) and _ASCII_ALNUM.match(tok[0]):
            parts.append(' ')
            pos += 1
        spans.append((pos, pos + len(tok)))
        parts.append(tok)
        pos += len(tok)
    return ''.join(parts), spans


def join_cjk(tokens):
    return join_cjk_spans(tokens)[0]
