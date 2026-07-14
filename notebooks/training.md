# MICH Training: Objectives, Losses, and Optimisation


## 1. The Fundamental Training Challenge

Before describing the individual loss terms, it is worth being precise about the constraint the model must eventually satisfy, and about how the current training recipe relates to that constraint, because the gap between the two shapes every design decision that follows.

At inference time on real data, we observe only BOLD signal, and we have no access to the ground-truth latent states $x$, $s$, $f$, $v$, $q$, $v^*$, $q^*$ that generated those observations. So whatever lets the model recover the latents at inference time has to come from BOLD alone: the data loss (predicted latents reproduce the observed BOLD through the forward model) and the physics loss (predicted latents satisfy the Heinzle ODEs), both described below, are the two objectives that do not require ground-truth latent trajectories and are the ones the model must ultimately be able to rely on by itself.

A separate caveat applies to both of them as currently implemented: the data loss's dedicated source-voxel term and the physics/data losses' biased collocation sampling (Section 4) all take the known location of the source voxel as an input, not just the BOLD signal. That is a different kind of privileged information from a ground-truth latent trajectory (a location, not a value over time), but it is still something a real, unlabelled scan would not hand you directly; it would need to come from elsewhere, for instance a task localiser or an externally chosen ROI. The one loss term that needs neither ground-truth latent values nor a known source location is the quiescence-consistency loss (Section 6), which is computed purely from the model's own predictions over the whole grid.

Training as currently configured does not rely on the data and physics losses alone. Synthetic data carries known ground-truth latents, and the training objective makes substantial, direct use of them: a supervision loss regresses the predicted latents against the true ones at the source voxel, at a weight comparable to the data loss itself, and a derivative-supervision term does the same for the analytic time derivative (Section 7). This is real latent supervision. It means the current training recipe, as configured today, depends on ground-truth latents that will not exist for real fMRI data; that dependency is exactly what motivates the questions in Sections 6, 7, and 10 about how much of it, and of the source-location dependency above, can eventually be removed.

This is a considerably harder problem than supervised regression, even setting the latent-supervision question aside. The BOLD signal underdetermines the latents: many combinations of $v$ and $q$ could reproduce a given BOLD trace. The physics constraints narrow this set substantially, but do not eliminate ambiguity entirely. Understanding this tension is essential to understanding why the training procedure is designed the way it is.

Synthetic data with known ground-truth latents also serves as a held-out evaluation oracle: the same latents used for supervision during training are compared directly against the model's validation-time predictions, since real fMRI offers no equivalent ground truth to validate against.

---

## 2. Input Normalisation

Before the BOLD signal is passed to the network, it is normalised to have approximately zero mean and unit variance. This is important for two reasons. First, the absolute amplitude of the BOLD signal varies across subjects, sessions, and acquisition parameters, and a network trained on unnormalised signals would not generalise well. Second, the physics loss involves ODE residuals whose scale depends on the signal amplitude, and normalised inputs keep those residuals in a numerically tractable range.

Normalisation is performed by a Welford online estimator, which computes running mean and variance statistics across training batches without storing all past data. The key design decision is *which voxels contribute to these statistics*. A naive approach would compute statistics across the entire spatial grid. This is problematic because the majority of voxels in any given scan are background: they carry little signal and their near-zero values would dominate the variance estimate, causing the source voxels (which carry meaningful hemodynamic signal) to be inappropriately rescaled.

Instead, statistics are computed from a spatial neighbourhood of a fixed radius around each sample's source voxel. This neighbourhood contains the voxels most likely to carry meaningful signal. A single scalar mean and variance is estimated from all voxels in this neighbourhood across all layers, time points, and batch elements. Using a shared scalar rather than per-layer statistics is a deliberate choice: it preserves inter-layer amplitude ratios, which carry information about the relative hemodynamic response across cortical depth.

The statistics are frozen after a fixed number of training steps (a step count set in config, not a detected convergence point). At validation and inference time, the frozen statistics from training are applied. The normalised signal is clamped to the range $[-10, 10]$ to prevent extreme values from destabilising training.

---

## 3. The Overall Loss

The total training loss is a weighted sum of several terms:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{data}} \cdot \mathcal{L}_{\text{data}} + \lambda_{\text{physics}}(t) \cdot \mathcal{L}_{\text{physics}} + \lambda_{\text{source-act}}(t) \cdot \mathcal{L}_{\text{source-act}} + \lambda_{\text{quiescence}}(t) \cdot \mathcal{L}_{\text{quiescence}} + \lambda_{\text{supervision}}(t) \cdot \mathcal{L}_{\text{supervision}}$$

where $t$ here denotes the training step rather than physical time. The $\lambda$ coefficients control the relative weight of each term, and those marked with $(t)$ are scheduled rather than fixed: they are ramped up from zero over the course of training for reasons discussed in Section 8. The data loss is the only term that is never scheduled; it is active at full weight from step 0.

Each term plays a distinct role. The data loss anchors the predicted latents to the observed BOLD signal (Section 4). The physics loss enforces consistency with the Heinzle ODEs (Section 5). The source-activity and quiescence-consistency losses are a pair of auxiliary terms that guard against two different ways the model can collapse to an uninformative solution: predicting flat, constant neural activity even at the labelled source, or hallucinating spurious activity away from it (Section 6). The supervision loss provides direct signal about the latent states at the source voxel, using the synthetic ground truth (Section 7).

Two further, opt-in terms sit alongside these but are not part of the formula above for brevity: a derivative-supervision loss (`supervise_dzdt`, on by default in the  config) that supervises the network's analytic $d\hat{z}/dt$ against an analytic ODE derivative computed from ground-truth latents, and an x-phase loss (`supervise_x_phase`, off by default) that adds a shape/phase-sensitive penalty on the recovered neural activity. Both are described briefly in Section 7 alongside the base supervision loss, since they share its dependence on ground truth. A smoothness term (`lambda_smooth`) also exists in the loss config but is disabled (weight 0) in the  configs.

---

## 4. The Data Loss

**What problem does this solve?**

The data loss is the primary anchor of training. Without it, the network has no incentive to pay attention to the observed BOLD signal at all: it could satisfy the physics loss by predicting the ODE resting-state fixed point everywhere, producing spatially uniform latents that evolve in a physically consistent but uninformative way. The data loss prevents this by requiring that the predicted latents, when passed through the BOLD forward model, reproduce the observed signal.

**The BOLD forward model.**

Given predicted blood volume $\hat{v}$ and deoxyhaemoglobin $\hat{q}$ at a voxel, the predicted BOLD signal is:

$$\hat{y} = V_0 \left[ k_1 (1 - \hat{q}) + k_2 \left(1 - \frac{\hat{q}}{\hat{v}}\right) + k_3 (1 - \hat{v}) \right]$$

where $V_0$, $k_1$, $k_2$, $k_3$ are acquisition-specific constants derived from the Heinzle model. This expression is differentiable with respect to $\hat{v}$ and $\hat{q}$, so gradients flow back through it to the network weights.

**The point spread function.**

The BOLD signal as measured by the scanner is not the ideal voxel-level hemodynamic response: spatial blurring from the acquisition process means that each voxel's measurement contains contributions from neighbouring voxels. This blurring is modelled by a Gaussian point spread function (PSF) applied independently to each cortical layer. After computing $\hat{y}$ from the predicted latents, the PSF is applied as a 2D convolution with a layer-specific Gaussian kernel before comparing to the observed signal. This step is essential for a fair comparison: without it, the loss would penalise the model for failing to match spatial structure that the scanner itself has blurred away.

**Collocation sampling.**

Computing the BOLD loss at every voxel and every time point for every training sample would be computationally prohibitive at the spatial and temporal resolution used here. Instead, a subset of space-time points is sampled at each training step. These are called *collocation points*, and the loss is computed only at these locations.

The sampling can be biased on both the spatial and temporal axes independently, each controlled by its own dense fraction: a configurable share of collocation points is drawn from a spatial neighbourhood around the known source voxel, and separately, a configurable share is drawn from a time window when the hemodynamic response is expected to be active, with the rest drawn uniformly across the full grid or the full duration respectively. When multiple sources are labelled in the same sample, the dense draws are round-robined across them, so no single source dominates the biased points. In the  configs, the spatial bias is active (95% of points drawn near the source) but the temporal bias is currently switched off (its dense fraction is set to 0), so collocation time points are drawn uniformly across the full duration rather than concentrated in a response window. The mechanism supports a biased time window and is exercised by the test suite, but it is not the  operating point.

**Source voxel loss.**

In addition to the collocation loss, a separate term computes the BOLD reconstruction error at the source voxel across the full time series. This term is given its own weight $\lambda_{\text{src}}$ and provides a dense, targeted signal at the location most relevant to the inversion. The total data loss is:

$$\mathcal{L}_{\text{data}} = \mathcal{L}_{\text{colloc}} + \lambda_{\text{src}} \cdot \mathcal{L}_{\text{src}}$$

**Failure mode without this term.** If the data loss is removed, the network collapses to the ODE resting-state fixed point: all predicted latents sit at their equilibrium values, the physics loss is satisfied trivially, and nothing is recovered. The data loss is therefore the minimum necessary condition for the inversion to be non-trivial.

---

## 5. The Physics Loss

**What problem does this solve?**

The data loss alone is underdetermined. Many combinations of $v$ and $q$ can reproduce a given BOLD trace, and the network has no reason to prefer combinations that are physically consistent over those that are not. The physics loss enforces consistency with the Heinzle ODEs, dramatically narrowing the space of valid solutions.

**What are the ODEs?**

The Heinzle model describes the temporal evolution of the hemodynamic state through a system of coupled first-order ODEs. Informally, neural activity $x$ drives a vasodilatory signal $s$, which in turn drives blood inflow $f$. Blood inflow drives changes in blood volume $v$ and deoxyhaemoglobin $q$. For datasets with a draining-vein compartment, $v$ and $q$ are additionally influenced by the corresponding states of the deeper layer ($v^*$, $q^*$). Each equation takes the form:

$$\frac{d z_i}{dt} = g_i(z_1, \ldots, z_N; \theta)$$

where $\theta$ denotes the haemodynamic parameters (time constants, coupling strengths, and so on) and $g_i$ is a known nonlinear function specific to each state variable.

The physics loss asks: do the predicted state trajectories $\hat{z}(t)$ actually satisfy these equations? It computes the ODE residual for each equation:

$$\mathcal{L}_{\text{phys}, i} = \left\| \frac{d\hat{z}_i}{dt} - g_i(\hat{z}_1, \ldots, \hat{z}_N; \theta) \right\|^2$$

where $\frac{d\hat{z}_i}{dt}$ is the analytic time derivative computed through the FiLM pathway (described in `heinzlenet.md`), and $g_i(\hat{z})$ is the ODE right-hand side evaluated at the predicted states. The ODE for $x$ is never included, since $x$ is the free driving input, not a variable with its own ODE. Whether $v^*$ and $q^*$ have their own ODE residuals depends on the dataset: datasets with a draining-vein compartment contribute six non-neural equations ($s, f, v, q, v^*, q^*$); single-layer datasets without one contribute four ($s, f, v, q$). The total physics loss averages these residuals across all applicable equations, all layers, and all collocation points.

**Numerical stabilisation.**

The ODE right-hand sides involve divisions and fractional exponents that can produce numerical instabilities when the predicted states take extreme values, particularly early in training. Before evaluating the right-hand sides, the predicted states are sanitised: NaN and infinity values are replaced with sensible defaults, and the positive-definite variables $f$, $v$, and $q$ are clamped to a minimum of $0.1$ to prevent division by near-zero values or undefined fractional powers. This sanitisation is applied only for the physics loss computation, not to the states returned by the network.

**The burn-in period.**

The ODE residuals can be evaluated only after a short burn-in of initial time steps, discarding a fixed number of initial collocation time indices before computing the residual. This exists because the network's predictions near $t = 0$ are unreliable: the temporal encoder has seen very little past context, and the network has not yet established a consistent estimate of the initial state. Including these early time points in the physics loss would introduce noisy gradients. The burn-in length is a config value (`burn_in`), currently set to 0 in the  configs, so this is presently a no-op; the mechanism exists for scenarios where it proves useful.

**Gradient computation.** The decoder's analytic time derivatives (needed for every ODE residual above) are only computed when the model actually requests them; requesting them unconditionally would waste compute whenever nothing needs them. In practice, though, they are computed on nearly every training step in the  configuration, since the derivative-supervision loss (`supervise_dzdt`, Section 7) also needs them and is on by default. See Section 8 for exactly which conditions trigger this.

**Failure mode without this term.** Without the physics loss, the network can produce latent trajectories that reproduce BOLD perfectly but are physically impossible: $v$ and $q$ could oscillate arbitrarily, $s$ could be uncoupled from $x$, and the inversion would have no physical interpretation. The physics loss is what gives the recovered latents their meaning as hemodynamic quantities.

---

## 6. The Source Activity and Quiescence Consistency Losses

**What problem do these solve?**

The data and physics losses alone leave two distinct collapse modes available to the network, both of which satisfy those losses while recovering nothing useful.

The first is collapse at the source itself: the network can predict a flat, constant $x$ at the labelled source voxel and still keep the data loss low, if the resulting BOLD is close enough to the observed signal on average. The **source-activity loss** guards against this. It computes the variance of the predicted $x$ over time at each labelled source voxel and penalises it whenever that variance falls below a small threshold $\epsilon$:

$$\mathcal{L}_{\text{source-act}} = \frac{1}{|\text{valid sources}|}\sum_{\text{valid } (b,s)} \max(0,\ \epsilon - \mathrm{Var}_t[\hat{x}_{b,s}(t)])$$

averaged only over the valid sources in a batch (samples can have a variable number of sources, padded slots are excluded via a mask). This is a hinge penalty: it costs nothing once the variance clears $\epsilon$, and only pushes back against a genuinely flat prediction.

The second is hallucination away from the source: the network can predict spurious, non-zero neural activity at background voxels that have no real source, since the data and physics losses are both evaluated at collocation points heavily biased toward the known source (Section 4) and rarely constrain the rest of the grid. The **quiescence-consistency loss** guards against this using a self-consistency argument rather than a fixed exclusion radius around each source. The physics loss's own $s$-equation residual implies $ds/dt = x - \kappa s - \gamma(f-1)$; wherever the model's own predicted $s$ and $f$ sit at their resting baseline ($s \approx 0$, $f \approx 1$) for a stretch of time, that residual being small forces $x \approx 0$ there too. The quiescence-consistency loss re-applies this same implied constraint, but densely, over every voxel and timestep, not just the sparse collocation points where the physics loss is actually evaluated:

$$\text{quiescent}(b,\ell,t,h,w) = \big(|\hat{s}| < \tau_s\big) \wedge \big(|\hat{f}-1| < \tau_f\big)$$
$$\mathcal{L}_{\text{quiescence}} = \frac{1}{|\text{quiescent voxels}|}\sum_{\text{quiescent}} \max(0,\ |\hat{x}| - \epsilon_x)$$

Both $\tau_s, \tau_f$ (how close to baseline counts as "quiescent") and $\epsilon_x$ (how much residual $|\hat{x}|$ is tolerated there) are generic numerical tolerances, not physics constants tied to any particular simulation scenario.

**Why this design, and not something simpler.** An earlier version of this idea used a single loss (informally, "antisteady") that combined two things: a variance-at-source term similar to the one above, and a separate geometric term that penalised activity in a fixed radius around each known source, treating everything outside that radius as presumed-quiescent. That geometric term was abandoned for two reasons. First, the correct radius is scenario-dependent, tied to the neural simulator's diffusion and decay constants, so there is no single safe value across simulation configs. Second, even a correctly-measured radius left very little of the grid penalisable once several sources were active in the same layer (a Monte Carlo check found roughly a 9% chance of zero penalisable coverage at all with four simultaneous sources on this project's grid). A related cross-layer term, which penalised activity at the same spatial column in every layer other than the source's own, was also dropped: it assumed a source's own layer was the only layer allowed to be active at that column, which is false whenever a scenario deliberately places sources at a shared position across layers.

The quiescence-consistency loss avoids both problems by keying off the model's own $s$ and $f$ predictions rather than a fixed geometric radius or an assumption about which layers can be active where. It needs no scenario-specific constant, it is naturally time-resolved (a voxel that is quiescent early but genuinely activated later, for instance by diffusion, is exempted the moment its own $s$/$f$ move off baseline), and it never runs out of coverage regardless of how many sources are active or how close together they sit.

**Status and scheduling.** Both losses are scheduled with the same delay-then-linear-warmup mechanism as the physics loss (Section 8), and independently of each other. The source-activity loss is on by default (its weight is non-zero out of the box, with no delay or warmup currently configured). The quiescence-consistency loss is opt-in (its weight defaults to 0) and, when enabled, is meant to be delayed until well after the physics loss has ramped up: enabling it from step 0 would reward the model for sitting at the trivial $s=0, f=1$ baseline everywhere, including at real sources, since that baseline is exactly where every signal starts before the data and physics losses have shaped any real dynamics. The specific delay/warmup values currently configured for the quiescence-consistency loss are a placeholder chosen only to fall after the physics loss's own delay and warmup, not a value validated against real training curves; treat it as a starting point to tune, not a settled default.

---

## 7. The Supervision Loss

The supervision loss applies a regression loss between the predicted and ground-truth latent states at the source voxel, across all layers and the full time series, using synthetic ground truth available only during training. It is applied to $s$, $f$, $v$, $q$, and, for datasets with a draining-vein compartment, $v^*$ and $q^*$. Note that $x$ is not directly supervised by this base term: the neural activity is the quantity the model is ultimately trying to recover, and supervising it directly would bypass the inversion entirely rather than test it.

The supervision is restricted to the source voxel rather than the full spatial grid. This reflects the fact that only one voxel per training sample has a known active neural source; supervising background voxels would provide misleading signal since those voxels should be at or near resting state.

This term was originally introduced as a diagnostic scaffold, to establish that the architecture and physics loss were capable of recovering the correct latents before asking whether the unsupervised signal (data loss plus physics loss) could do so alone. In the current  configuration it is not scheduled to fade out: it carries a substantial, non-decaying weight, comparable to the data loss itself, active from step 0. Two further terms have since been built on top of it, both also active during training rather than purely diagnostic:

- **Derivative supervision** (`supervise_dzdt`, on by default) supervises the network's analytic $d\hat{z}/dt$ against an ODE derivative computed from the ground-truth latents, at the source voxel. This is more sensitive to timing and phase than the value-only supervision above, which matters because $x$ itself is never directly supervised and is only ever recovered through the physics residual against the (well-supervised) $s$ trajectory. In principle this target is meant to use ground-truth $s, f, v, q$ but not ground-truth $x$, so that it stays applicable in settings without ground-truth neural activity; in the current implementation, the analytic target for the $s$-equation derivative is computed as $\dot s = x_{\text{true}} - \kappa s_{\text{true}} - \gamma(f_{\text{true}}-1)$, which does read ground-truth $x$ directly. This is a real gap between the stated intent and the current code, worth checking before relying on this term's independence from ground-truth neural activity.
- **X-phase supervision** (`supervise_x_phase`, opt-in) adds a shape- and phase-sensitive penalty (a combination of MSE and a Pearson-correlation term) directly on the recovered neural activity at the source voxel. Its own internal MSE/Pearson balance, and its overall weight, can optionally be annealed over a configured step window rather than held fixed, so that its influence on the total gradient does not fade late in training as other terms grow.

There is also a separate, explicitly temporary ablation switch, `supervise_x`, off by default, that directly supervises $x$ against ground truth. Unlike the terms above, this one is documented in code as existing only to diagnose whether certain neural activity patterns are hard to learn because of the architecture rather than because of identifiability, and is not part of the normal training objective.

---

## 8. Loss Scheduling and the Warm-Up Strategy

**Why scheduling is necessary.**

The loss terms do not all play well together at every stage of training. The most dangerous interaction is between the data loss and the physics loss early in training. At initialisation, the network's predictions are essentially random: the latents have no meaningful structure, and the BOLD reconstruction is poor. If the physics loss is applied at full strength from the beginning, it will dominate the gradient signal and drive the network toward the ODE resting-state fixed point, where all variables sit at their equilibrium values. This satisfies the physics loss trivially but destroys the data loss signal. Once the network has converged to the fixed point, escaping it is very difficult.

The solution is to delay and ramp the physics loss. During the early phase of training, only the data loss is active. The network is free to find any solution that roughly reproduces the BOLD signal. Once the data loss has established a reasonable initialisation, the physics loss is introduced gradually, so that the network can adapt its predictions toward physical consistency without losing the BOLD signal entirely.

**How scheduling is implemented.**

Each scheduled loss term has two parameters: a *delay* and a *warm-up duration*. During the delay period, the effective weight of that term is exactly zero. After the delay, the weight is linearly ramped from zero to its target value over the warm-up duration. Formally:

$$\lambda(t) = \begin{cases} 0 & t < t_{\text{delay}} \\ \lambda_{\text{target}} \cdot \min\left(1,\, \frac{t - t_{\text{delay}}}{t_{\text{warmup}}}\right) & t \geq t_{\text{delay}} \end{cases}$$

where $t$ is the global training step. Every scheduled term (physics, source-activity, quiescence-consistency, supervision, derivative-supervision, x-phase) uses this same delay-then-linear-warmup mechanism through one shared helper; only the delay/warmup values differ per term, several of them currently 0 (meaning that term, if enabled, applies its target weight from step 0 with no ramp). The data loss is the one exception: it is active from the beginning and is not scheduled at all.

An important implementation detail: the time derivative computation in the decoder, needed for the physics loss, is expensive. The original motivation for gating it was to avoid that cost while the physics loss weight is still zero during its delay period. In the  default configuration, though, this gate rarely actually skips the computation: derivatives are also requested whenever validation is running, or whenever the derivative-supervision loss (`supervise_dzdt`) or the x-phase loss (`supervise_x_phase`) are enabled, and `supervise_dzdt` is on by default. In practice this means derivatives are computed on nearly every training step, not just once the physics loss has ramped up.

---

## 9. The Validation Procedure

At validation time, the model is evaluated on held-out synthetic data with known ground-truth latents. This is the closest available proxy to an oracle evaluation: we can directly compare the predicted latent trajectories against the true ones without any ambiguity.

Validation runs the full forward pass including time derivatives, unconditionally, regardless of the physics loss schedule, since the schedule is a training device and validation should always reflect the full model capability. The same losses as in training are computed and logged, but the primary diagnostic is the visual comparison of predicted and ground-truth trajectories at the source voxel.

Two sets of plots are generated at the end of each validation epoch. The first compares predicted and true BOLD signals alongside predicted and true neural activity $x$ across all layers, for a random subset of validation samples. The second compares the full set of predicted and true latent trajectories ($s$, $f$, $v$, $q$, and $v^*$, $q^*$ where applicable) in the same format. These plots are logged to Weights and Biases and serve as the primary qualitative indicator of training progress, alongside quantitative recovery metrics (R², Pearson correlation, peak cross-correlation lag) computed over the full validation set.

---

## 10. The Training Objective as a Whole

It is worth pausing to appreciate the structure of what has been described. The network is being asked to invert a nonlinear dynamical system from its outputs alone, without direct observation of its internal states. Two principled constraints define this as an optimisation problem: find latent trajectories that (a) reproduce the observed BOLD when passed through the forward model, and (b) satisfy the governing ODEs.

Each constraint alone is insufficient. Constraint (a) alone is underdetermined: many latent trajectories can reproduce a given BOLD trace. Constraint (b) alone has a trivial solution at the resting-state fixed point, where all variables sit at equilibrium and nothing is recovered. Together, they narrow the solution space considerably, but not all the way: the source-activity and quiescence-consistency losses (Section 6) close two further collapse routes that (a) and (b) leave open, a flat prediction at the source and unconstrained hallucination away from it, without requiring ground-truth latents to do so.

The supervision loss and its derivative/phase-aligned extensions (Section 7) add a further constraint that anchors the solution at the source voxel using synthetic ground truth, currently present as a standing, substantially-weighted part of the  objective rather than a term scheduled to fade out. The longer-term goal for the project remains a model that recovers the latent states from constraints (a) and (b), plus the source-activity and quiescence-consistency guards, without any ground-truth latent supervision at all; whether that unsupervised signal is sufficient on its own, without the current supervision terms, is the open question the supervision loss was originally introduced to help answer, and remains the harder question this training setup is working toward.

The warm-up scheduling exists because the interaction between the data and physics losses is fragile early in training. The ordering matters: the data loss must first establish a non-trivial solution, then the physics loss can refine it toward physical consistency. Violating this ordering by introducing the physics loss too early tends to collapse training to the resting-state fixed point, from which recovery is very difficult. Understanding this ordering is as important as understanding the individual loss terms themselves.
