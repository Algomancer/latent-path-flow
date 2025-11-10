Thought about this a little more.


tldr: 

OU posterior q_phi(c | x) using soft time-local attention:
  - GRU -> h_seq[B,T,H], h_global[B,H]
  - compute attn_k = Σ_j w_{k,j} * h_seq_j around each t_k
  - build ctx_k = concat(h_global + attn_k, t_k)
  - produce per-step OU params (mu_k, kappa_k, sigma_k) from ctx_k
  - q(c_0 | x): from h_global + attn_0

