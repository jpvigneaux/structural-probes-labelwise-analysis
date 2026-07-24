"""Classes for training and running inference on probes."""
import os
import sys

from torch import optim
import torch
import torch.nn.functional as F
from tqdm import tqdm

class ProbeRegimen:
  """Basic regimen for training and running inference on probes.
  
  Tutorial help from:
  https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

  Attributes:
    optimizer: the optimizer used to train the probe
    scheduler: the scheduler used to set the optimizer base learning rate
  """

  def __init__(self, args):
    self.args = args
    self.max_epochs = args['probe_training']['epochs']
    self.params_path = os.path.join(args['reporting']['root'], args['probe']['params_path'])

  def set_optimizer(self, probe):
    """Sets the optimizer and scheduler for the training regimen.
  
    Args:
      probe: the probe PyTorch model the optimizer should act on.
    """
    self.optimizer = optim.Adam(probe.parameters(), lr=0.001)
    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1,patience=0)

  def train_until_convergence(self, probe, model, loss, train_dataset, dev_dataset, reporter=None):
    """ Trains a probe until a convergence criterion is met.

    Trains until loss on the development set does not improve by more than epsilon
    for 5 straight epochs.

    Writes parameters of the probe to disk, at the location specified by config.

    Args:
      probe: An instance of probe.Probe, transforming model outputs to predictions
      model: An instance of model.Model, transforming inputs to word reprs
      loss: An instance of loss.Loss, computing loss between predictions and labels
      train_dataset: a torch.DataLoader object for iterating through training data
      dev_dataset: a torch.DataLoader object for iterating through dev data
    """
    self.set_optimizer(probe)
    min_dev_loss = sys.maxsize
    min_dev_loss_epoch = -1
    for epoch_index in tqdm(range(self.max_epochs), desc='[training]'):
      epoch_train_loss = 0
      epoch_dev_loss = 0
      epoch_train_epoch_count = 0
      epoch_dev_epoch_count = 0
      epoch_train_loss_count = 0
      epoch_dev_loss_count = 0
      for batch in tqdm(train_dataset, desc='[training batch]'):
        probe.train()
        self.optimizer.zero_grad()
        observation_batch, label_batch, length_batch, _ = batch
        word_representations = model(observation_batch)
        predictions = probe(word_representations)
        batch_loss, count = loss(predictions, label_batch, length_batch)
        batch_loss.backward()
        epoch_train_loss += batch_loss.detach().cpu().numpy()*count.detach().cpu().numpy()
        epoch_train_epoch_count += 1
        epoch_train_loss_count += count.detach().cpu().numpy()
        self.optimizer.step()
      epoch_dev_predictions = [] if reporter is not None else None
      epoch_dev_batches = [] if reporter is not None else None
      for batch in tqdm(dev_dataset, desc='[dev batch]'):
        self.optimizer.zero_grad()
        probe.eval()
        observation_batch, label_batch, length_batch, _ = batch
        word_representations = model(observation_batch)
        predictions = probe(word_representations)
        batch_loss, count = loss(predictions, label_batch, length_batch)
        epoch_dev_loss += batch_loss.detach().cpu().numpy()*count.detach().cpu().numpy()
        epoch_dev_loss_count += count.detach().cpu().numpy()
        epoch_dev_epoch_count += 1
        if reporter is not None:
          epoch_dev_predictions.append(predictions.detach().cpu().numpy())
          epoch_dev_batches.append(batch)
      self.scheduler.step(epoch_dev_loss)
      tqdm.write('[epoch {}] Train loss: {}, Dev loss: {}'.format(epoch_index, epoch_train_loss/epoch_train_loss_count, epoch_dev_loss/epoch_dev_loss_count))
      if reporter is not None and hasattr(reporter, 'report_training_epoch'):
        reporter.report_training_epoch(
            epoch_index, epoch_dev_predictions, epoch_dev_batches,
            epoch_train_loss / epoch_train_loss_count,
            epoch_dev_loss / epoch_dev_loss_count)
      if epoch_dev_loss / epoch_dev_loss_count < min_dev_loss - 0.0001:
        torch.save(probe.state_dict(), self.params_path)
        min_dev_loss = epoch_dev_loss / epoch_dev_loss_count
        min_dev_loss_epoch = epoch_index
        tqdm.write('Saving probe parameters')
      elif min_dev_loss_epoch < epoch_index - 4:
        tqdm.write('Early stopping')
        break

  def predict(self, probe, model, dataset):
    """ Runs probe to compute predictions on a dataset.

    Args:
      probe: An instance of probe.Probe, transforming model outputs to predictions
      model: An instance of model.Model, transforming inputs to word reprs
      dataset: A pytorch.DataLoader object

    Returns:
      A list of predictions for each batch in the batches yielded by the dataset
    """
    probe.eval()
    predictions_by_batch = []
    for batch in tqdm(dataset, desc='[predicting]'):
      observation_batch, label_batch, length_batch, _ = batch
      word_representations = model(observation_batch)
      predictions = probe(word_representations)
      predictions_by_batch.append(predictions.detach().cpu().numpy())
    return predictions_by_batch


class PolarRegimen(ProbeRegimen):
  """Training regimen for PolarProbe.

  Extends ProbeRegimen to additionally extract (h_head − h_dep) edge vectors
  and their dependency-type IDs from each batch, projecting them through
  probe.forward_edges() for the angular loss component.

  The rel_vocab (relation string → integer) is built lazily from training data
  and reused for dev/test.
  """

  def __init__(self, args):
    super().__init__(args)
    self.rel_vocab = {}

  def _extract_edges(self, word_reps, raw_batch):
    """Collect projected-edge inputs for L_A from a batch.

    For each gold dependency edge (head→dep) in the batch sentences, computes
    the difference vector (h_head − h_dep) and its integer relation-type ID.

    Skips the synthetic root attachment (head_index = 0 in CoNLL-X).

    Args:
      word_reps: (B, max_len, hidden) float tensor
      raw_batch:  list of (observation, label) tuples returned by the dataloader

    Returns:
      edge_diffs: (E, hidden) float tensor, or None if no valid edges
      edge_types: (E,) LongTensor of relation IDs, or None
    """
    diffs, types = [], []
    for b, (obs, _) in enumerate(raw_batch):
      for dep_i, (head_str, rel) in enumerate(
          zip(obs.head_indices, obs.governance_relations)):
        try:
          head_j = int(head_str) - 1   # 0-indexed; -1 = root
        except ValueError:
          continue
        if head_j < 0:
          continue
        if rel not in self.rel_vocab:
          self.rel_vocab[rel] = len(self.rel_vocab)
        diffs.append(word_reps[b, head_j] - word_reps[b, dep_i])
        types.append(self.rel_vocab[rel])

    if not diffs:
      return None, None
    edge_diffs = torch.stack(diffs)                               # (E, hidden)
    edge_types = torch.tensor(types, device=word_reps.device)    # (E,)
    return edge_diffs, edge_types

  def _run_batch(self, probe, model, loss_fn, batch, train=True):
    obs_batch, label_batch, length_batch, raw_batch = batch
    word_reps   = model(obs_batch)
    dist_preds  = probe(word_reps)
    edge_diffs, edge_types = self._extract_edges(word_reps, raw_batch)
    edge_embs   = probe.forward_edges(edge_diffs) if edge_diffs is not None else None
    predictions = (dist_preds, edge_embs, edge_types)
    return loss_fn(predictions, label_batch, length_batch)

  def train_until_convergence(self, probe, model, loss_fn, train_dataset,
                              dev_dataset, reporter=None):
    self.set_optimizer(probe)
    min_dev_loss       = sys.maxsize
    min_dev_loss_epoch = -1

    for epoch_index in tqdm(range(self.max_epochs), desc='[training]'):
      epoch_train_loss = epoch_dev_loss = 0
      epoch_train_count = epoch_dev_count = 0

      for batch in tqdm(train_dataset, desc='[training batch]'):
        probe.train()
        self.optimizer.zero_grad()
        batch_loss, count = self._run_batch(probe, model, loss_fn, batch, train=True)
        batch_loss.backward()
        epoch_train_loss  += batch_loss.detach().cpu().item() * count.detach().cpu().item()
        epoch_train_count += count.detach().cpu().item()
        self.optimizer.step()

      epoch_dev_predictions = [] if reporter is not None else None
      epoch_dev_batches     = [] if reporter is not None else None
      for batch in tqdm(dev_dataset, desc='[dev batch]'):
        probe.eval()
        self.optimizer.zero_grad()
        batch_loss, count = self._run_batch(probe, model, loss_fn, batch, train=False)
        epoch_dev_loss  += batch_loss.detach().cpu().item() * count.detach().cpu().item()
        epoch_dev_count += count.detach().cpu().item()
        if reporter is not None:
          obs_batch, _, _, _ = batch
          word_reps  = model(obs_batch)
          dist_preds = probe(word_reps)
          epoch_dev_predictions.append(dist_preds.detach().cpu().numpy())
          epoch_dev_batches.append(batch)

      self.scheduler.step(epoch_dev_loss)
      train_avg = epoch_train_loss / max(epoch_train_count, 1)
      dev_avg   = epoch_dev_loss   / max(epoch_dev_count,   1)
      tqdm.write(f'[epoch {epoch_index}] Train loss: {train_avg:.6f}, Dev loss: {dev_avg:.6f}')

      if reporter is not None and hasattr(reporter, 'report_training_epoch'):
        reporter.report_training_epoch(epoch_index, epoch_dev_predictions,
                                       epoch_dev_batches, train_avg, dev_avg)

      if dev_avg < min_dev_loss - 0.0001:
        torch.save(probe.state_dict(), self.params_path)
        min_dev_loss       = dev_avg
        min_dev_loss_epoch = epoch_index
        tqdm.write('Saving probe parameters')
      elif min_dev_loss_epoch < epoch_index - 4:
        tqdm.write('Early stopping')
        break
