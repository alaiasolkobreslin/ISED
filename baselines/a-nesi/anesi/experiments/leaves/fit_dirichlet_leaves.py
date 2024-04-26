import torch
from torch.distributions import Dirichlet
from torch.nn.functional import softplus


def fit_dirichlet(beliefs:list[torch.Tensor], alpha, optimizer, iters=1000, L2=0.0) -> Dirichlet:
    """
    Fit a Dirichlet distribution to the beliefs.
    :param beliefs: Tensor of shape (K, |W|, n) where K is the number of beliefs, |W| is number of elements in a world
     and n is the number of classes.
    :param alpha: Tensor of shape (|W|, n) of the prior.
    :param lr: Learning rate for alpha
    :param iters: Number of iterations to optimize log-probability of Dirichlet
    :param L2: L2 regularization on alpha. If 0, no regularization. If > 0, this will prefer lower values of alpha.
     This is used to prevent the Dirichlet distribution from becoming too peaked on the uniform distribution over classes
    """

    a1 = softplus(alpha[0])
    a2 = softplus(alpha[1])
    a3 = softplus(alpha[2])
    
    [data1, data2, data3] = beliefs
    N = len(beliefs)
    eps = 10e-8
    statistics1 = (data1 + eps).log().mean(0).detach()
    statistics2 = (data2 + eps).log().mean(0).detach()
    statistics3 = (data3 + eps).log().mean(0).detach()
    for i in range(iters):


        # Dirichlet log likelihood. See https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf
        # statistics = data.log().mean(0)
        # NOTE: The hardcoded version is quicker since the sufficient statistics are fixed.
        # NOTE: We have to think about how to parallelize this. Should be trivial (especially if the dimensions of each dirichlet is fixed)
        t1 = torch.lgamma(a1.sum(-1) + eps) - torch.lgamma(a1 + eps).sum(-1) + torch.sum((a1 - 1) * statistics1, -1)
        t2 = torch.lgamma(a2.sum(-1) + eps) - torch.lgamma(a2 + eps).sum(-1) + torch.sum((a2 - 1) * statistics2, -1)
        t3 = torch.lgamma(a3.sum(-1) + eps) - torch.lgamma(a3 + eps).sum(-1) + torch.sum((a3 - 1) * statistics3, -1)
        log_p = t1 + t2 + t3
        
        # log_p = log_p * N
        optimizer.zero_grad()

        a = ((a1**2).sum() + (a2**2).sum() + (a3**2).sum())/(len(a1) + len(a2) + len(a3))
        loss = -(log_p) + L2 * a

        loss.backward(retain_graph=True)
        optimizer.step()
        # Sometimes the algorithm will find negative numbers during minimizing the log probability.
        # However alpha needs to be positive.
    
        a1 = softplus(alpha[0])
        a2 = softplus(alpha[1])
        a3 = softplus(alpha[2])

    return Dirichlet(a1), Dirichlet(a2), Dirichlet(a3)