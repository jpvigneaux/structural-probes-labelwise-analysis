"""Contains classes for computing and reporting evaluation metrics."""

from collections import defaultdict
import os

from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
import numpy as np 
import json

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
mpl.rcParams['agg.path.chunksize'] = 10000

class Reporter:
  """Base class for reporting.

  Attributes:
    test_reporting_constraint: Any reporting method
      (identified by a string) not in this list will not
      be reported on for the test set.
  """

  def __init__(self, args):
    raise NotImplementedError("Inherit from this class and override __init__")

  def __call__(self, prediction_batches, dataloader, split_name):
    """
    Performs all reporting methods as specifed in the yaml experiment config dict.
    
    Any reporting method not in test_reporting_constraint will not
      be reported on for the test set.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataloader: A DataLoader for a data split
      split_name the string naming the data split: {train,dev,test}
    """
    for method in self.reporting_methods:
      if method in self.reporting_method_dict:
        if split_name == 'test' and method not in self.test_reporting_constraint:
          tqdm.write("Reporting method {} not in test set reporting "
              "methods (reporter.py); skipping".format(method))
          continue
        tqdm.write("Reporting {} on split {}".format(method, split_name))
        self.reporting_method_dict[method](prediction_batches
            , dataloader, split_name)
      else:
        tqdm.write('[WARNING] Reporting method not known: {}; skipping'.format(method))

  def write_json(self, prediction_batches, dataset, split_name):
    """Writes observations and predictions to disk.
    
    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    json.dump([prediction_batch.tolist() for prediction_batch in prediction_batches]
        , open(os.path.join(self.reporting_root, split_name+'.predictions'), 'w'))
    json.dump([[x[0][:-1] for x in observation_batch] for _,_,_, observation_batch in dataset],
        open(os.path.join(self.reporting_root, split_name+'.observations'), 'w'))

class WordPairReporter(Reporter):
  """Reporting class for wordpair (distance) tasks"""

  def __init__(self, args):
    self.args = args
    self.reporting_methods = args['reporting']['reporting_methods']
    self.reporting_method_dict = {
        'spearmanr': self.report_spearmanr,
        'image_examples':self.report_image_examples,
        'uuas':self.report_uuas_and_tikz,
        'uuas_by_relation':self.report_uuas_by_relation,
        'write_predictions':self.write_json
        }
    self.reporting_root = args['reporting']['root']
    self.test_reporting_constraint = {'spearmanr', 'uuas', 'uuas_by_relation', 'root_acc'}

  def report_spearmanr(self, prediction_batches, dataset, split_name):
    """Writes the Spearman correlations between predicted and true distances.

    For each word in each sentence, computes the Spearman correlation between
    all true distances between that word and all other words, and all
    predicted distances between that word and all other words.

    Computes the average such metric between all sentences of the same length.
    Writes these averages to disk.
    Then computes the average Spearman across sentence lengths 5 to 50;
    writes this average to disk.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    lengths_to_spearmanrs = defaultdict(list)
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(
        prediction_batches, dataset):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        words = observation.sentence
        length = int(length)
        prediction = prediction[:length,:length]
        label = label[:length,:length].cpu()
        spearmanrs = [spearmanr(pred, gold) for pred, gold in zip(prediction, label)]
        lengths_to_spearmanrs[length].extend([x.correlation for x in spearmanrs])
    mean_spearman_for_each_length = {length: np.mean(lengths_to_spearmanrs[length]) 
        for length in lengths_to_spearmanrs}

    with open(os.path.join(self.reporting_root, split_name + '.spearmanr'), 'w') as fout:
      for length in sorted(mean_spearman_for_each_length):
        fout.write(str(length) + '\t' + str(mean_spearman_for_each_length[length]) + '\n')

    with open(os.path.join(self.reporting_root, split_name + '.spearmanr-5_50-mean'), 'w') as fout:
      mean = np.mean([mean_spearman_for_each_length[x] for x in range(5,51) if x in mean_spearman_for_each_length])
      fout.write(str(mean) + '\n')

  def report_image_examples(self, prediction_batches, dataset, split_name):
    """Writes predicted and gold distance matrices to disk for the first 20
    elements of the developement set as images!

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    images_printed = 0
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(
        prediction_batches, dataset):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        length = int(length)
        prediction = prediction[:length,:length]
        label = label[:length,:length].cpu()
        words = observation.sentence
        fontsize = 5*( 1 + np.sqrt(len(words))/200)
        plt.clf()
        ax = sns.heatmap(label)
        ax.set_title('Gold Parse Distance')
        ax.set_xticks(np.arange(len(words)))
        ax.set_yticks(np.arange(len(words)))
        ax.set_xticklabels(words, rotation=90, fontsize=fontsize, ha='center')
        ax.set_yticklabels(words, rotation=0, fontsize=fontsize, va='top')
        plt.tight_layout()
        plt.savefig(os.path.join(self.reporting_root, split_name + '-gold'+str(images_printed)), dpi=300)

        plt.clf()
        ax = sns.heatmap(prediction)
        ax.set_title('Predicted Parse Distance (squared)')
        ax.set_xticks(np.arange(len(words)))
        ax.set_yticks(np.arange(len(words)))
        ax.set_xticklabels(words, rotation=90, fontsize=fontsize, ha='center')
        ax.set_yticklabels(words, rotation=0, fontsize=fontsize, va='center')
        plt.tight_layout()
        plt.savefig(os.path.join(self.reporting_root, split_name + '-pred'+str(images_printed)), dpi=300)
        print('Printing', str(images_printed))
        images_printed += 1
        if images_printed == 20:
          return

  def report_training_epoch(self, epoch_index, prediction_batches, dev_batches, train_loss, dev_loss):
    """Logs per-epoch UUAS and per-relation UUAS on the dev set during training.

    Appends one row to training_uuas.tsv (epoch, train_loss, dev_loss, dev_uuas)
    and one row per relation to training_uuas_by_relation.tsv
    (epoch, relation, uuas, correct, total).

    Args:
      epoch_index: integer index of the current epoch
      prediction_batches: list of numpy arrays of predictions collected during
          the dev pass of this epoch (same format as regimen.predict output)
      dev_batches: list of raw DataLoader batches collected during the dev pass,
          each a tuple (data_batch, label_batch, length_batch, observation_batch)
      train_loss: scalar training loss for this epoch
      dev_loss: scalar dev loss for this epoch
    """
    uspan_correct = 0
    uspan_total = 0
    relation_correct = defaultdict(int)
    relation_total = defaultdict(int)

    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(
        prediction_batches, dev_batches):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        words = observation.sentence
        poses = observation.xpos_sentence
        length = int(length)
        prediction = prediction[:length, :length]
        label = label[:length, :length].cpu()

        gold_edges = prims_matrix_to_edges(label, words, poses)
        pred_edges = prims_matrix_to_edges(prediction, words, poses)
        pred_edges_set = set([tuple(sorted(x)) for x in pred_edges])

        uspan_correct += len(
            set([tuple(sorted(x)) for x in gold_edges]).intersection(pred_edges_set))
        uspan_total += len(gold_edges)

        edge_to_relation = {}
        for token_idx in range(length):
          head_idx = int(observation.head_indices[token_idx]) - 1
          if head_idx < 0 or head_idx >= length:
            continue
          edge = tuple(sorted((token_idx, head_idx)))
          edge_to_relation[edge] = observation.governance_relations[token_idx]

        for i, j in gold_edges:
          edge = tuple(sorted((i, j)))
          relation = edge_to_relation.get(edge, 'UNK')
          relation_total[relation] += 1
          if edge in pred_edges_set:
            relation_correct[relation] += 1

    uuas = uspan_correct / float(uspan_total) if uspan_total > 0 else 0.0

    uuas_path = os.path.join(self.reporting_root, 'training_uuas.tsv')
    write_header = not os.path.exists(uuas_path)
    with open(uuas_path, 'a') as fout:
      if write_header:
        fout.write('epoch\ttrain_loss\tdev_loss\tdev_uuas\n')
      fout.write('{}\t{:.6f}\t{:.6f}\t{:.4f}\n'.format(
          epoch_index, train_loss, dev_loss, uuas))

    rel_path = os.path.join(self.reporting_root, 'training_uuas_by_relation.tsv')
    write_header = not os.path.exists(rel_path)
    with open(rel_path, 'a') as fout:
      if write_header:
        fout.write('epoch\trelation\tuuas\tcorrect\ttotal\n')
      for relation in sorted(relation_total.keys()):
        total = relation_total[relation]
        correct = relation_correct[relation]
        fout.write('{}\t{}\t{:.4f}\t{}\t{}\n'.format(
            epoch_index, relation, correct / float(total), correct, total))

  def report_uuas_and_tikz(self, prediction_batches, dataset, split_name):
    """Computes the UUAS score for a dataset and writes tikz dependency latex.

    From the true and predicted distances, computes a minimum spanning tree
    of each, and computes the percentage overlap between edges in all
    predicted and gold trees.

    For the first 20 examples (if not the test set) also writes LaTeX to disk
    for visualizing the gold and predicted minimum spanning trees.

    All tokens with punctuation part-of-speech are excluded from the minimum
    spanning trees.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    uspan_total = 0
    uspan_correct = 0
    total_sents = 0
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in tqdm(zip(
        prediction_batches, dataset), desc='[uuas,tikz]'):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        words = observation.sentence
        poses = observation.xpos_sentence
        length = int(length)
        assert length == len(observation.sentence)
        prediction = prediction[:length,:length]
        label = label[:length,:length].cpu()

        gold_edges = prims_matrix_to_edges(label, words, poses)
        pred_edges = prims_matrix_to_edges(prediction, words, poses)

        if split_name != 'test' and total_sents < 20:
          self.print_tikz(pred_edges, gold_edges, words, split_name)

        uspan_correct += len(set([tuple(sorted(x)) for x in gold_edges]).intersection(
          set([tuple(sorted(x)) for x in pred_edges])))
        uspan_total += len(gold_edges)
        total_sents += 1
    uuas = uspan_correct / float(uspan_total)
    with open(os.path.join(self.reporting_root, split_name + '.uuas'), 'w') as fout:
      fout.write(str(uuas) + '\n')

  def report_uuas_by_relation(self, prediction_batches, dataset, split_name):
    """Computes UUAS broken down by dependency relation type.

    For each dependency relation label (e.g., nsubj, obj, det), computes
    the fraction of gold edges of that type that are recovered in the
    predicted minimum spanning tree. Writes a TSV to
    {split_name}.uuas_by_relation with columns: relation, uuas, correct, total.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name: the string naming the data split: {train,dev,test}
    """
    relation_correct = defaultdict(int)
    relation_total = defaultdict(int)

    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in tqdm(zip(
        prediction_batches, dataset), desc='[uuas_by_relation]'):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        words = observation.sentence
        poses = observation.xpos_sentence
        length = int(length)
        prediction = prediction[:length, :length]
        label = label[:length, :length].cpu()

        gold_edges = prims_matrix_to_edges(label, words, poses)
        pred_edges = prims_matrix_to_edges(prediction, words, poses)
        pred_edges_set = set([tuple(sorted(x)) for x in pred_edges])

        # Map each undirected tree edge to its relation label using
        # head_indices (1-indexed in CoNLL; 0 = root) and governance_relations
        edge_to_relation = {}
        for token_idx in range(length):
          head_idx = int(observation.head_indices[token_idx]) - 1  # 0-indexed
          if head_idx < 0 or head_idx >= length:  # root token has no incoming edge
            continue
          edge = tuple(sorted((token_idx, head_idx)))
          edge_to_relation[edge] = observation.governance_relations[token_idx]

        for i, j in gold_edges:
          edge = tuple(sorted((i, j)))
          relation = edge_to_relation.get(edge, 'UNK')
          relation_total[relation] += 1
          if edge in pred_edges_set:
            relation_correct[relation] += 1

    with open(os.path.join(self.reporting_root, split_name + '.uuas_by_relation'), 'w') as fout:
      fout.write('relation\tuuas\tcorrect\ttotal\n')
      for relation in sorted(relation_total.keys()):
        total = relation_total[relation]
        correct = relation_correct[relation]
        uuas = correct / float(total)
        fout.write('{}\t{:.4f}\t{}\t{}\n'.format(relation, uuas, correct, total))

  def print_tikz(self, prediction_edges, gold_edges, words, split_name):
    ''' Turns edge sets on word (nodes) into tikz dependency LaTeX. '''
    with open(os.path.join(self.reporting_root, split_name+'.tikz'), 'a') as fout:
      string = """\\begin{dependency}[hide label, edge unit distance=.5ex]
    \\begin{deptext}[column sep=0.05cm]
    """ 
      string += "\\& ".join([x.replace('$', '\\$').replace('&', '+') for x in words]) + " \\\\" + '\n'
      string += "\\end{deptext}" + '\n'
      for i_index, j_index in gold_edges:
        string += '\\depedge{{{}}}{{{}}}{{{}}}\n'.format(i_index+1,j_index+1, '.')
      for i_index, j_index in prediction_edges:
        string += '\\depedge[edge style={{red!60!}}, edge below]{{{}}}{{{}}}{{{}}}\n'.format(i_index+1,j_index+1, '.')
      string += '\\end{dependency}\n'
      fout.write('\n\n')
      fout.write(string)

class WordReporter(Reporter):
  """Reporting class for single-word (depth) tasks"""

  def __init__(self, args):
    self.args = args
    self.reporting_methods = args['reporting']['reporting_methods']
    self.reporting_method_dict = {
        'spearmanr': self.report_spearmanr,
        'root_acc':self.report_root_acc,
        'write_predictions':self.write_json,
        'image_examples':self.report_image_examples,
        }
    self.reporting_root = args['reporting']['root']
    self.test_reporting_constraint = {'spearmanr', 'uuas', 'root_acc'}

  def report_spearmanr(self, prediction_batches, dataset, split_name):
    """Writes the Spearman correlations between predicted and true depths.

    For each sentence, computes the spearman correlation between predicted
    and true depths.

    Computes the average such metric between all sentences of the same length.
    Writes these averages to disk.
    Then computes the average Spearman across sentence lengths 5 to 50;
    writes this average to disk.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    lengths_to_spearmanrs = defaultdict(list)
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(
        prediction_batches, dataset):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        words = observation.sentence
        length = int(length)
        prediction = prediction[:length]
        label = label[:length].cpu()
        sent_spearmanr = spearmanr(prediction, label)
        lengths_to_spearmanrs[length].append(sent_spearmanr.correlation)
    mean_spearman_for_each_length = {length: np.mean(lengths_to_spearmanrs[length]) 
        for length in lengths_to_spearmanrs}

    with open(os.path.join(self.reporting_root, split_name + '.spearmanr'), 'w') as fout:
      for length in sorted(mean_spearman_for_each_length):
        fout.write(str(length) + '\t' + str(mean_spearman_for_each_length[length]) + '\n')

    with open(os.path.join(self.reporting_root, split_name + '.spearmanr-5_50-mean'), 'w') as fout:
      mean = np.mean([mean_spearman_for_each_length[x] for x in range(5,51) if x in mean_spearman_for_each_length])
      fout.write(str(mean) + '\n')

  def report_root_acc(self, prediction_batches, dataset, split_name):
    """Computes the root prediction accuracy and writes to disk.

    For each sentence in the corpus, the root token in the sentence
    should be the least deep. This is a simple evaluation.

    Computes the percentage of sentences for which the root token
    is the least deep according to the predicted depths; writes
    this value to disk.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    total_sents = 0
    correct_root_predictions = 0
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(
        prediction_batches, dataset):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        length = int(length)
        label = list(label[:length].cpu())
        prediction = prediction.data[:length]
        words = observation.sentence
        poses = observation.xpos_sentence

        correct_root_predictions += label.index(0) == get_nopunct_argmin(prediction, words, poses)
        total_sents += 1

    root_acc = correct_root_predictions / float(total_sents)
    with open(os.path.join(self.reporting_root, split_name + '.root_acc'), 'w') as fout:
      fout.write('\t'.join([str(root_acc), str(correct_root_predictions), str(total_sents)]) + '\n')

  def report_image_examples(self, prediction_batches, dataset, split_name):
    """Writes predicted and gold depths to disk for the first 20
    elements of the developement set as images!

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    images_printed = 0
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(
        prediction_batches, dataset):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        plt.clf()
        length = int(length)
        prediction = prediction[:length]
        label = label[:length].cpu()
        words = observation.sentence
        fontsize = 6
        cumdist = 0
        for index, (word, gold, pred) in enumerate(zip(words, label, prediction)):
          plt.text(cumdist*3, gold*2, word, fontsize=fontsize, ha='center')
          plt.text(cumdist*3, pred*2, word, fontsize=fontsize, color='red', ha='center')
          cumdist = cumdist + (np.square(len(word)) + 1)

        plt.ylim(0,20)
        plt.xlim(0,cumdist*3.5)
        plt.title('LSTM H Encoder Dependency Parse Tree Depth Prediction', fontsize=10)
        plt.ylabel('Tree Depth', fontsize=10)
        plt.xlabel('Linear Absolute Position',fontsize=10)
        plt.tight_layout()
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)
        plt.savefig(os.path.join(self.reporting_root, split_name + '-depth'+str(images_printed)), dpi=300)
        images_printed += 1
        if images_printed == 20:
          return


class UnionFind:
  '''
  Naive UnionFind implementation for (slow) Prim's MST algorithm

  Used to compute minimum spanning trees for distance matrices
  '''
  def __init__(self, n):
    self.parents = list(range(n))
  def union(self, i,j):
    if self.find(i) != self.find(j):
      i_parent = self.find(i)
      self.parents[i_parent] = j
  def find(self, i):
    i_parent = i
    while True:
      if i_parent != self.parents[i_parent]:
        i_parent = self.parents[i_parent]
      else:
        break
    return i_parent


def prims_matrix_to_edges(matrix, words, poses):
  '''
  Constructs a minimum spanning tree from the pairwise weights in matrix;
  returns the edges.

  Never lets punctuation-tagged words be part of the tree.
  '''
  pairs_to_distances = {}
  uf = UnionFind(len(matrix))
  for i_index, line in enumerate(matrix):
    for j_index, dist in enumerate(line):
      if poses[i_index] in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]:
        continue
      if poses[j_index] in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]:
        continue
      pairs_to_distances[(i_index, j_index)] = dist
  edges = []
  for (i_index, j_index), distance in sorted(pairs_to_distances.items(), key = lambda x: x[1]):
    if uf.find(i_index) != uf.find(j_index):
      uf.union(i_index, j_index)
      edges.append((i_index, j_index))
  return edges

def get_nopunct_argmin(prediction, words, poses):
  '''
  Gets the argmin of predictions, but filters out all punctuation-POS-tagged words
  '''
  puncts = ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]
  original_argmin = np.argmin(prediction)
  for i in range(len(words)):
    argmin = np.argmin(prediction)
    if poses[argmin] not in puncts:
      return argmin
    else:
      prediction[argmin] = 9000
  return original_argmin
