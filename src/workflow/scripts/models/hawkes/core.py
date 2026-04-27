"""
Core mathematical engine for fitting univariate continuous-time
Hawkes processes with an exponentially decaying excitation kernel
and a user-defined piecewise-constant background intensity.
"""

import numpy as np
from scipy.optimize import minimize

def compute_Ah(t, alpha, beta):
    """
    Computes the recurrent excitation term for each event.
    A(i) = sum_{j < i} alpha * exp(-beta * (t_i - t_j))
    
    Computed in O(N) using the recursion:
    A(i) = exp(-beta * (t_i - t_{i-1})) * (A(i-1) + alpha)
    
    Args:
        t: array of event timestamps (sorted natively)
        alpha: jump size
        beta: decay rate
        
    Returns:
        array A of length N
    """
    N = len(t)
    A = np.zeros(N)
    if N == 0:
        return A
    
    for i in range(1, N):
        dt = t[i] - t[i-1]
        A[i] = np.exp(-beta * dt) * (A[i-1] + alpha)
        
    return A

def neg_loglik_hawkes(params, t, mu_base_t, M_base_T, T):
    """
    Computes the negative log-likelihood of the Hawkes process.
    
    Args:
        params: array [kappa, alpha, beta]
        t: array of N event timestamps
        mu_base_t: array of N background intensities at each t_i (normalized so integral is 1 over T)
        M_base_T: scalar, integral of mu_base over [0, T]. If mu_base is normalized, this is 1.0.
        T: end of the observation window
        
    Returns:
        float: -log L
    """
    kappa, alpha, beta = params
    
    # Force parameters to be strictly positive to avoid math domain errors
    kappa = max(kappa, 1e-10)
    alpha = max(alpha, 1e-10)
    beta = max(beta, 1e-10)
    
    N = len(t)
    if N == 0:
        return kappa * M_base_T
    
    # 1. Compensator (Integral of intensity over [0, T])
    # Integral of background
    comp_bg = kappa * M_base_T
    
    # Integral of excitations
    comp_ex = (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - t)))
    
    Lambda_T = comp_bg + comp_ex
    
    # 2. Sum of log intensities at event times
    A = compute_Ah(t, alpha, beta)
    lambda_at_t = kappa * mu_base_t + A
    
    # Ensure strict positivity robustly without destroying gradient completely
    lambda_at_t = np.maximum(lambda_at_t, 1e-10)
        
    log_sum = np.sum(np.log(lambda_at_t))
    
    logL = log_sum - Lambda_T
    # We want to MINIMIZE negative logL
    return -logL

def neg_loglik_hawkes_reparam(params, t, mu_base_t, M_base_T, T):
    kappa, br, beta = params
    alpha = br * beta
    return neg_loglik_hawkes([kappa, alpha, beta], t, mu_base_t, M_base_T, T)

def fit_hawkes(t, mu_base_t, M_base_T, T, init_params=None):
    """
    Fits the Hawkes process to timestamps t.
    
    Args:
        t: array of event timestamps (sorted)
        mu_base_t: the normalized background intensity evaluated at each t_i
        M_base_T: total integral of background intensity over [0, T]
        T: total observation window
        init_params: [kappa, alpha, beta]
        
    Returns:
        dict with fitted parameters and optimizer success flag
    """
    N = len(t)
    
    if init_params is None:
        # Heuristic initialization
        # kappa starts assuming 50% of events are background
        kappa0 = max(0.01, (0.5 * N) / M_base_T) if M_base_T > 0 else 1.0
        # alpha, beta initialized to reasonable transit values (e.g. beta = decay in ~5 mins, alpha < beta)
        # Assuming t is in minutes, 5 mins decay -> beta = 1/5 = 0.2
        beta0 = 0.2
        alpha0 = 0.1
        init_params = np.array([kappa0, alpha0, beta0])
        
    kappa_init, alpha_init, beta_init = init_params
    br_init = min(0.99, max(1e-4, alpha_init / beta_init))
    reparam_init = np.array([kappa_init, br_init, beta_init])
        
    bounds = [
        (1e-5, None),      # kappa > 0
        (1e-5, 0.999),     # branching_ratio < 1 (stability)
        (1e-5, None)       # beta > 0
    ]
    
    res = minimize(
        neg_loglik_hawkes_reparam, 
        reparam_init, 
        args=(t, mu_base_t, M_base_T, T),
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    
    kappa_opt, br_opt, beta_opt = res.x
    alpha_opt = br_opt * beta_opt
    
    return {
        'kappa': kappa_opt,
        'alpha': alpha_opt,
        'beta': beta_opt,
        'branching_ratio': br_opt,
        'loglik': -res.fun,
        'converged': res.success,
        'message': res.message
    }

def compute_compensator_tau(params, t, M_base_t):
    """
    Computes the transformed times tau_i = Lambda(t_i) for time-rescaling Goodness-of-Fit.

    O(N) implementation via the recursion:
        B[i] = Σ_{j<i} exp(-β(t_i - t_j)) = exp(-β·Δt_i) · (B[i-1] + 1)
    which is identical to compute_Ah(t, 1.0, beta).  The excitation integral
    then collapses to:
        Σ_{j<i} (α/β)(1 - exp(-β(t_i-t_j)))  =  (α/β) · (i - B[i])

    Args:
        params: [kappa, alpha, beta]
        t: array of N event timestamps (sorted)
        M_base_t: array of N values, integral of background intensity from 0 to each t_i.

    Returns:
        tau: array of N transformed times
    """
    kappa, alpha, beta = params
    N = len(t)
    if N == 0:
        return np.zeros(0)

    # B[i] = Σ_{j<i} exp(-β(t_i - t_j)) — reuse compute_Ah with jump=1
    B = compute_Ah(t, 1.0, beta)

    i_vals = np.arange(N, dtype=float)
    ex = (alpha / beta) * (i_vals - B)

    tau = kappa * M_base_t + ex
    return tau

def simulate_hawkes_branching(params, profile, rng=None):
    """
    Simulates a continuous-time Hawkes process on [0, T] using the branching representation.
    This generates exact events bypassing Ogata thinning rejections.
    
    Args:
        params: [kappa, alpha, beta]
        profile: dict with keys "mu_blocks" (rates), "dt_sec", "T_total"
        rng: np.random.Generator (optional)
        
    Returns:
        array of exactly simulated sorted timestamps
    """
    if rng is None:
        rng = np.random.default_rng()
        
    kappa, alpha, beta = params
    mu_blocks = profile["mu_blocks"]
    dt_sec = profile["dt_sec"]
    T_total = profile["T_total"]
    
    # 1. Background Process Simulation
    events = []
    for i, mu_k in enumerate(mu_blocks):
        bg_rate = kappa * mu_k
        expected_bg = bg_rate * dt_sec
        if expected_bg > 0:
            n_bg = rng.poisson(expected_bg)
            t_start = i * dt_sec
            t_end = min((i + 1) * dt_sec, T_total)
            if n_bg > 0:
                t_bg = rng.uniform(t_start, t_end, size=n_bg)
                events.extend(t_bg)
                
    events = np.array(events)
    all_events = [events]
    
    # 2. Branching Process Simulation
    queue = events
    branching_ratio = alpha / beta
    
    while len(queue) > 0:
        # Number of offspring for each event in queue
        n_offsprings = rng.poisson(branching_ratio, size=len(queue))
        
        # Calculate exactly how many offspring to generate total
        total_off = np.sum(n_offsprings)
        if total_off == 0:
            break
            
        # Repeat the parent times based on how many offspring they generated
        # e.g., if queue=[t1, t2] and n_offsprings=[2, 0], parents=[t1, t1]
        parents = np.repeat(queue, n_offsprings)
        
        # Draw exponential offsets
        offsets = rng.exponential(scale=1.0/beta, size=total_off)
        
        new_times = parents + offsets
        
        # Keep only events inside the window
        valid = new_times <= T_total
        new_queue = new_times[valid]
        
        if len(new_queue) > 0:
            all_events.append(new_queue)
            queue = new_queue
        else:
            break
            
    if not all_events:
        return np.array([])
        
    final_events = np.concatenate(all_events)
    final_events.sort()
    return final_events
