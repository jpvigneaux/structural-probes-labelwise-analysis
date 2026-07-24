"""Custom loss classes for probing tasks."""

import torch
import torch.nn as nn


class RankingDistanceLoss(nn.Module):
  """Pairwise margin ranking loss for distance matrices.

  For every ordered pair of word-pairs (a, b) within a sentence where
  gold_distance_a < gold_distance_b, penalises:

      max(0,  pred_a − pred_b + margin)

  The loss is scale-invariant: only the ordering of predicted distances
  matters, not their absolute values.  This is the natural companion to
  TwoWordOrthogonalProbe (cosine-based distances in [0, 2]) and is
  directly aligned with both evaluation metrics:
    - UUAS  (Prim's MST depends only on distance ranking)
    - Spearman correlation  (rank-based by definition)

  Per-sentence loss is normalised by the number of ordered pairs so that
  sentence length does not dominate; the batch loss is the mean over sentences.

  Config keys (all under probe_training):
    margin  (float, default 0.0): minimum required gap between pred_a and pred_b.
  """
  def __init__(self, args):
    super(RankingDistanceLoss, self).__init__()
    self.args = args
    self.margin = float(args['probe_training'].get('margin', 0.0))

  def forward(self, predictions, label_batch, length_batch):
    """
    Args:
      predictions:  (B, max_len, max_len) predicted distances
      label_batch:  (B, max_len, max_len) gold distances; -1 for padding
      length_batch: (B,) actual sentence lengths

    Returns:
      batch_loss:   scalar mean loss over non-empty sentences
      total_sents:  number of non-empty sentences (for upstream averaging)
    """
    total_sents = torch.sum(length_batch != 0).float()
    batch_loss  = torch.tensor(0.0, device=self.args['device'])

    for b in range(predictions.size(0)):
      L = int(length_batch[b].item())
      if L < 2:
        continue

      pred_sq = predictions[b, :L, :L]          # (L, L)
      gold_sq = label_batch[b, :L, :L]          # (L, L)

      # Upper triangle, excluding padding (gold == -1)
      tri  = torch.triu(torch.ones(L, L, dtype=torch.bool,
                                   device=predictions.device), diagonal=1)
      mask = tri & (gold_sq != -1)

      p = pred_sq[mask]        # (M,) predicted distances
      g = gold_sq[mask].float()  # (M,) gold distances

      M = p.size(0)
      if M < 2:
        continue

      # (M, M) matrix of gold-distance differences; entry [a,b] = g[a] − g[b]
      gold_diff = g.unsqueeze(1) - g.unsqueeze(0)
      ordered   = (gold_diff < 0).float()   # 1 where g[a] < g[b]

      n_ordered = ordered.sum()
      if n_ordered == 0:
        continue

      # Penalise pred_a > pred_b (+ margin) when g[a] < g[b]
      pred_diff  = p.unsqueeze(1) - p.unsqueeze(0)   # p[a] − p[b]
      violations = (pred_diff + self.margin).clamp(min=0)

      sent_loss  = (violations * ordered).sum() / n_ordered
      batch_loss = batch_loss + sent_loss

    if total_sents > 0:
      batch_loss = batch_loss / total_sents

    return batch_loss, total_sents

class PolarLoss(nn.Module):
  """Combined structural + angular loss for the Polar Probe.

  L_total = L_S + λ · L_A

  L_S  — L1 loss on predicted parse distances (same as L1DistanceLoss).
  L_A  — pairwise contrastive cosine loss over directly-connected edge pairs:
          L_A = mean_{(s,s')} ( cos(Bs, Bs') − 1[type(s)=type(s')] )²
          Pushes same-type projected edges to be collinear (target cos=1)
          and different-type edges to be orthogonal (target cos=0).

  The forward() signature expects *predictions* to be a 3-tuple:
      (dist_predictions, edge_embeddings, edge_type_ids)
  where dist_predictions is the usual (B, N, N) distance matrix,
  edge_embeddings is (E, rank) or None (if no edges in batch),
  and edge_type_ids is a LongTensor of shape (E,) or None.

  Config keys (under probe_training):
    lambda_angular  (float, default 10.0): weight on L_A.
  """
  def __init__(self, args):
    super(PolarLoss, self).__init__()
    self.args = args
    self.lambda_angular = float(args['probe_training'].get('lambda_angular', 10.0))
    self.word_pair_dims = (1, 2)

  def forward(self, predictions, label_batch, length_batch):
    dist_preds, edge_embeddings, edge_type_ids = predictions

    # ── L_S: L1 on predicted distances ────────────────────────────────────
    labels_1s = (label_batch != -1).float()
    predictions_masked = dist_preds * labels_1s
    labels_masked      = label_batch * labels_1s
    total_sents   = torch.sum((length_batch != 0)).float()
    squared_lengths = length_batch.pow(2).float()
    if total_sents > 0:
      loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked),
                                dim=self.word_pair_dims)
      L_S = torch.sum(loss_per_sent / squared_lengths) / total_sents
    else:
      L_S = torch.tensor(0.0, device=self.args['device'])

    # ── L_A: contrastive cosine loss on edge pairs ─────────────────────────
    if edge_embeddings is None or edge_embeddings.shape[0] < 2:
      return L_S, total_sents

    norms    = edge_embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
    edge_n   = edge_embeddings / norms                    # (E, rank), unit vectors
    cos_mat  = torch.matmul(edge_n, edge_n.t())           # (E, E)

    same_type = (edge_type_ids.unsqueeze(1) ==
                 edge_type_ids.unsqueeze(0)).float()       # (E, E)

    E = edge_embeddings.shape[0]
    triu = torch.triu(torch.ones(E, E, device=edge_embeddings.device,
                                 dtype=torch.bool), diagonal=1)
    L_A = torch.mean((cos_mat[triu] - same_type[triu]) ** 2)

    return L_S + self.lambda_angular * L_A, total_sents


class L1DistanceLoss(nn.Module):
  """Custom L1 loss for distance matrices."""
  def __init__(self, args):
    super(L1DistanceLoss, self).__init__()
    self.args = args
    self.word_pair_dims = (1,2)

  def forward(self, predictions, label_batch, length_batch):
    """ Computes L1 loss on distance matrices.

    Ignores all entries where label_batch=-1
    Normalizes first within sentences (by dividing by the square of the sentence length)
    and then across the batch.

    Args:
      predictions: A pytorch batch of predicted distances
      label_batch: A pytorch batch of true distances
      length_batch: A pytorch batch of sentence lengths

    Returns:
      A tuple of:
        batch_loss: average loss in the batch
        total_sents: number of sentences in the batch
    """
    labels_1s = (label_batch != -1).float()
    predictions_masked = predictions * labels_1s
    labels_masked = label_batch * labels_1s
    total_sents = torch.sum((length_batch != 0)).float()
    squared_lengths = length_batch.pow(2).float()
    if total_sents > 0:
      loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=self.word_pair_dims)
      normalized_loss_per_sent = loss_per_sent / squared_lengths
      batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
    else:
      batch_loss = torch.tensor(0.0, device=self.args['device'])
    return batch_loss, total_sents


class L1DepthLoss(nn.Module):
  """Custom L1 loss for depth sequences."""
  def __init__(self, args):
    super(L1DepthLoss, self).__init__()
    self.args = args
    self.word_dim = 1

  def forward(self, predictions, label_batch, length_batch):
    """ Computes L1 loss on depth sequences.

    Ignores all entries where label_batch=-1
    Normalizes first within sentences (by dividing by the sentence length)
    and then across the batch.

    Args:
      predictions: A pytorch batch of predicted depths
      label_batch: A pytorch batch of true depths
      length_batch: A pytorch batch of sentence lengths

    Returns:
      A tuple of:
        batch_loss: average loss in the batch
        total_sents: number of sentences in the batch
    """
    total_sents = torch.sum(length_batch != 0).float()
    labels_1s = (label_batch != -1).float()
    predictions_masked = predictions * labels_1s
    labels_masked = label_batch * labels_1s
    if total_sents > 0:
      loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=self.word_dim)
      normalized_loss_per_sent = loss_per_sent / length_batch.float()
      batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
    else:
      batch_loss = torch.tensor(0.0, device=self.args['device'])
    return batch_loss, total_sents
