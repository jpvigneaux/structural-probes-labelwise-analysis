"""Classes for specifying probe pytorch modules."""

import torch.nn as nn
import torch
import torch.nn.functional as F

class Probe(nn.Module):
  pass

init_range = 0.00005

class TwoWordPSDProbe(Probe):
  """ Computes squared L2 distance after projection by a matrix.

  For a batch of sentences, computes all n^2 pairs of distances
  for each sentence in the batch.
  """
  def __init__(self, args):
    print('Constructing TwoWordPSDProbe')
    super(TwoWordPSDProbe, self).__init__()
    self.args = args
    self.probe_rank = args['probe']['maximum_rank']
    self.model_dim = args['model']['hidden_dim']
    self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.probe_rank))
    nn.init.uniform_(self.proj, -init_range, init_range)
    self.to(args['device'])

  def forward(self, batch):
    """ Computes all n^2 pairs of distances after projection
    for each sentence in a batch.

    Note that due to padding, some distances will be non-zero for pads.
    Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
    """
    transformed = torch.matmul(batch, self.proj)
    batchlen, seqlen, rank = transformed.size()
    transformed = transformed.unsqueeze(2)
    transformed = transformed.expand(-1, -1, seqlen, -1)
    transposed = transformed.transpose(1,2)
    diffs = transformed - transposed
    squared_diffs = diffs.pow(2)
    squared_distances = torch.sum(squared_diffs, -1)
    return squared_distances


class TwoWordOrthogonalProbe(Probe):
  """Squared-L2 distance in a rank-r subspace with orthonormal basis (Stiefel manifold).

  For a batch of sentences, computes all n² pairwise distances:

      d(i,j) = ‖U^T(hᵢ − hⱼ)‖²  =  (hᵢ−hⱼ)^T UU^T (hᵢ−hⱼ)

  where U ∈ R^{d_model × r} has orthonormal columns.  This is the squared
  L2 distance *projected onto the learned subspace* — identical in spirit to
  TwoWordPSDProbe but restricted to the Stiefel manifold (isometric maps).
  Distances are non-negative and grow unboundedly with parse distance, so
  the ranking loss and MST both receive well-scaled signal.

  The orthonormality constraint is enforced via QR reparameterization:
  an unconstrained W is stored; U = Q from QR(W) at each forward pass.
  PyTorch autograd differentiates through torch.linalg.qr without hooks.

  Previous variant (archived below): used d(i,j) = 1 − cos(LN(Uhᵢ), LN(Uhⱼ)).
  LN forced all projected vectors to unit norm, making distances purely angular
  and bounded in [0, 2].  This compressed resolution for large parse distances
  and caused poor UUAS (≈58 % vs ≈81 % for TwoWordPSDProbe).  The L2² metric
  recovers full dynamic range while keeping the orthonormal-basis constraint.
  """
  def __init__(self, args):
    print('Constructing TwoWordOrthogonalProbe')
    super(TwoWordOrthogonalProbe, self).__init__()
    self.args = args
    self.probe_rank = args['probe']['maximum_rank']
    self.model_dim  = args['model']['hidden_dim']
    self.W = nn.Parameter(data=torch.empty(self.model_dim, self.probe_rank))
    nn.init.orthogonal_(self.W)
    self.to(args['device'])

  def forward(self, batch):
    """Pairwise squared-L2 distances in the orthonormal subspace.

    Args:
      batch: (batch_size, max_seq_len, d_model)
    Returns:
      distances: (batch_size, max_seq_len, max_seq_len), non-negative, unbounded
    """
    U, _ = torch.linalg.qr(self.W)                   # (d_model, probe_rank)
    projected = torch.matmul(batch, U)                # (B, N, probe_rank)
    batchlen, seqlen, rank = projected.size()
    proj_i = projected.unsqueeze(2).expand(-1, -1, seqlen, -1)
    diffs  = proj_i - proj_i.transpose(1, 2)
    return torch.sum(diffs.pow(2), -1)

  # ── archived cosine variant ────────────────────────────────────────────────
  # def forward_cosine(self, batch):
  #   """Original cosine-of-LN variant; distances ∈ [0, 2]."""
  #   U, _ = torch.linalg.qr(self.W)
  #   projected = torch.matmul(batch, U)
  #   p = F.layer_norm(projected, [self.probe_rank], weight=None, bias=None)
  #   cos_sim = torch.bmm(p, p.transpose(1, 2)) / self.probe_rank
  #   return 1.0 - cos_sim



class OneWordPSDProbe(Probe):
  """ Computes squared L2 norm of words after projection by a matrix."""

  def __init__(self, args):
    print('Constructing OneWordPSDProbe')
    super(OneWordPSDProbe, self).__init__()
    self.args = args
    self.probe_rank = args['probe']['maximum_rank']
    self.model_dim = args['model']['hidden_dim']
    self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.probe_rank))
    nn.init.uniform_(self.proj, -init_range, init_range)
    self.to(args['device'])

  def forward(self, batch):
    """ Computes all n depths after projection
    for each sentence in a batch.

    Computes (Bh_i)^T(Bh_i) for all i

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of depths of shape (batch_size, max_seq_len)
    """
    transformed = torch.matmul(batch, self.proj)
    batchlen, seqlen, rank = transformed.size()
    norms = torch.bmm(transformed.view(batchlen* seqlen, 1, rank),
        transformed.view(batchlen* seqlen, rank, 1))
    norms = norms.view(batchlen, seqlen)
    return norms

class OneWordNonPSDProbe(Probe):
  """Computes a bilinear affinity between each word representation and itself.
  
  This is different from the probes in A Structural Probe... as the
  matrix in the quadratic form is not guaranteed positive semi-definite
  
  """

  def __init__(self, args):
    print('Constructing OneWordNonPSDProbe')
    super(OneWordNonPSDProbe, self).__init__()
    self.args = args
    self.model_dim = args['model']['hidden_dim']
    self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.model_dim))
    nn.init.uniform_(self.proj, -init_range, init_range)
    self.to(args['device'])

  def forward(self, batch):
    """ Computes all n depths after projection
    for each sentence in a batch.

    Computes (h_i^T)A(h_i) for all i

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of depths of shape (batch_size, max_seq_len)
    """
    transformed = torch.matmul(batch, self.proj)
    batchlen, seqlen, rank = batch.size()
    norms = torch.bmm(transformed.view(batchlen* seqlen, 1, rank),
        batch.view(batchlen*seqlen, rank, 1))
    norms = norms.view(batchlen, seqlen)
    return norms

class PolarProbe(Probe):
  """Unconstrained linear projection for the Polar Probe (Diego-Simón et al., 2024).

  Architecture is identical to TwoWordPSDProbe (B ∈ R^{d_model × rank}, squared
  L2 distance for the structural objective) but exposes forward_edges() so that
  PolarRegimen can compute projected edge vectors for the angular objective:

      d̂(hᵢ, hⱼ)  = ‖(hᵢ − hⱼ) B‖²          (structural)
      ê(h_hd, h_dp) = (h_hd − h_dp) B          (angular; normalised → cosine)
  """
  def __init__(self, args):
    print('Constructing PolarProbe')
    super(PolarProbe, self).__init__()
    self.args = args
    self.probe_rank = args['probe']['maximum_rank']
    self.model_dim  = args['model']['hidden_dim']
    self.proj = nn.Parameter(data=torch.zeros(self.model_dim, self.probe_rank))
    nn.init.uniform_(self.proj, -init_range, init_range)
    self.to(args['device'])

  def forward(self, batch):
    """Pairwise squared-L2 distances in the probe subspace.

    Args:
      batch: (batch_size, max_seq_len, d_model)
    Returns:
      distances: (batch_size, max_seq_len, max_seq_len)
    """
    transformed = torch.matmul(batch, self.proj)      # (B, N, rank)
    batchlen, seqlen, rank = transformed.size()
    transformed = transformed.unsqueeze(2).expand(-1, -1, seqlen, -1)
    transposed  = transformed.transpose(1, 2)
    diffs = transformed - transposed
    return torch.sum(diffs.pow(2), -1)

  def forward_edges(self, edge_diffs):
    """Project edge-difference vectors for the angular objective.

    Args:
      edge_diffs: (E, d_model)  —  (h_head − h_dep) for each gold edge
    Returns:
      projected: (E, rank)
    """
    return torch.matmul(edge_diffs, self.proj)


class TwoWordNonPSDProbe(Probe):
  """ Computes a bilinear function of difference vectors.

  For a batch of sentences, computes all n^2 pairs of scores
  for each sentence in the batch.
  """
  def __init__(self, args):
    print('TwoWordNonPSDProbe')
    super(TwoWordNonPSDProbe, self).__init__()
    self.args = args
    self.probe_rank = args['probe']['maximum_rank']
    self.model_dim = args['model']['hidden_dim']
    self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.model_dim))
    nn.init.uniform_(self.proj, -init_range, init_range)
    self.to(args['device'])

  def forward(self, batch):
    """ Computes all n^2 pairs of difference scores 
    for each sentence in a batch.

    Note that due to padding, some distances will be non-zero for pads.
    Computes (h_i-h_j)^TA(h_i-h_j) for all i,j

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of scores of shape (batch_size, max_seq_len, max_seq_len)
    """
    batchlen, seqlen, rank = batch.size()
    batch_square = batch.unsqueeze(2).expand(batchlen, seqlen, seqlen, rank)
    diffs = (batch_square - batch_square.transpose(1,2)).view(batchlen*seqlen*seqlen, rank)
    psd_transformed = torch.matmul(diffs, self.proj).view(batchlen*seqlen*seqlen,1,rank)
    dists = torch.bmm(psd_transformed, diffs.view(batchlen*seqlen*seqlen, rank, 1))
    dists = dists.view(batchlen, seqlen, seqlen)
    return dists
