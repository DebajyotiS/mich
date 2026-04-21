# MICH Training: Objectives, Losses, and Optimisation


## 1. The Fundamental Training Challenge

Before describing the individual loss terms, it is worth being precise about the constraint under which training operates, because it shapes every design decision that follows.

At inference time, we observe only BOLD signal. We have no access to the ground-truth latent states $x$, $s$, $f$, $v$, $q$, $v^*$, $q^*$ that generated those observations. This means we cannot simply train by comparing the network's predicted latents against true latents at every voxel and time point. If we did, the model would depend on supervision that is unavailable when deployed on real data.

The training objective must therefore be *self-supervised with respect to the latents*: the network is trained only on signals that are available during real inference. The primary such signal is the BOLD observation itself. We train the network to produce latent states that are consistent with the observed BOLD in two complementary senses: they should *reproduce* the BOLD when passed through the forward hemodynamic model, and they should *satisfy* the Heinzle ODEs that govern how those states evolve over time.

This is a considerably harder problem than supervised regression. The BOLD signal underdetermines the latents: many combinations of $v$ and $q$ could reproduce a given BOLD trace. The physics constraints narrow this set substantially, but do not eliminate ambiguity entirely. Understanding this tension is essential to understanding why the training procedure is designed the way it is.

Synthetic data with known ground-truth latents is used during training for a secondary supervision signal and as a held-out evaluation oracle. Crucially, the ground-truth latents are never used in the primary training loss, only in an auxiliary term and for validation. This preserves the validity of the evaluation: a model that has never seen ground-truth latents during training can be fairly evaluated against them at validation time.

---

## 2. Input Normalisation

Before the BOLD signal is passed to the network, it is normalised to have approximately zero mean and unit variance. This is important for two reasons. First, the absolute amplitude of the BOLD signal varies across subjects, sessions, and acquisition parameters, and a network trained on unnormalised signals would not generalise well. Second, the physics loss involves ODE residuals whose scale depends on the signal amplitude, and normalised inputs keep those residuals in a numerically tractable range.

Normalisation is performed by a Welford online estimator, which computes running mean and variance statistics across training batches without storing all past data. The key design decision is *which voxels contribute to these statistics*. A naive approach would compute statistics across the entire spatial grid. This is problematic because the majority of voxels in any given scan are background: they carry little signal and their near-zero values would dominate the variance estimate, causing the source voxels (which carry meaningful hemodynamic signal) to be inappropriately rescaled.

Instead, statistics are computed from a spatial neighbourhood of a fixed radius around each sample's source voxel. This neighbourhood contains the voxels most likely to carry meaningful signal. A single scalar mean and variance is estimated from all voxels in this neighbourhood across all layers, time points, and batch elements. Using a shared scalar rather than per-layer statistics is a deliberate choice: it preserves inter-layer amplitude ratios, which carry information about the relative hemodynamic response across cortical depth.

Once the running statistics have converged, they are frozen after a fixed number of training steps. At validation and inference time, the frozen statistics from training are applied. The normalised signal is clamped to the range $[-10, 10]$ to prevent extreme values from destabilising training.

---

## 3. The Overall Loss

The total training loss is a weighted sum of three terms:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{data}} \cdot \mathcal{L}_{\text{data}} + \lambda_{\text{physics}}(t) \cdot \mathcal{L}_{\text{physics}} + \lambda_{\text{supervision}}(t) \cdot \mathcal{L}_{\text{supervision}}$$

where $t$ here denotes the training step rather than physical time. The $\lambda$ coefficients control the relative weight of each term, and those marked with $(t)$ are scheduled rather than fixed: they are ramped up from zero over the course of training for reasons discussed in Section 7.

Each term plays a distinct role. The data loss anchors the predicted latents to the observed BOLD signal. The physics loss enforces consistency with the Heinzle ODEs. The supervision loss provides direct signal about the latent states at the source voxel, using the synthetic ground truth. These three objectives serve specific purpose, as described below.

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

The sampling is not uniform. A fraction of collocation points are drawn densely from a spatial neighbourhood around the known source voxel and from the time window when the hemodynamic response is active ($5\%$ to $55\%$ of the total duration). The remaining points are drawn uniformly across space and time. This biased sampling ensures that the loss gradient is concentrated where the signal is most informative: near the source, during the response window.

**Source voxel loss.**

In addition to the collocation loss, a separate term computes the BOLD reconstruction error at the source voxel across the full time series. This term is given its own weight $\lambda_{\text{src}}$ and provides a dense, targeted signal at the location most relevant to the inversion. The total data loss is:

$$\mathcal{L}_{\text{data}} = \mathcal{L}_{\text{colloc}} + \lambda_{\text{src}} \cdot \mathcal{L}_{\text{src}}$$

**Failure mode without this term.** If the data loss is removed, the network collapses to the ODE resting-state fixed point: all predicted latents sit at their equilibrium values, the physics loss is satisfied trivially, and nothing is recovered. The data loss is therefore the minimum necessary condition for the inversion to be non-trivial.

---

## 5. The Physics Loss

**What problem does this solve?**

The data loss alone is underdetermined. Many combinations of $v$ and $q$ can reproduce a given BOLD trace, and the network has no reason to prefer combinations that are physically consistent over those that are not. The physics loss enforces consistency with the Heinzle ODEs, dramatically narrowing the space of valid solutions.

**What are the ODEs?**

The Heinzle model describes the temporal evolution of the hemodynamic state through a system of coupled first-order ODEs. Informally, neural activity $x$ drives a vasodilatory signal $s$, which in turn drives blood inflow $f$. Blood inflow drives changes in blood volume $v$ and deoxyhaemoglobin $q$, both of which are influenced by a draining-vein compartment from the deeper layer ($v^*$, $q^*$). Each equation takes the form:

$$\frac{d z_i}{dt} = g_i(z_1, \ldots, z_7; \theta)$$

where $\theta$ denotes the haemodynamic parameters (time constants, coupling strengths, and so on) and $g_i$ is a known nonlinear function specific to each state variable.

The physics loss asks: do the predicted state trajectories $\hat{z}(t)$ actually satisfy these equations? It computes the ODE residual for each equation:

$$\mathcal{L}_{\text{phys}, i} = \left\| \frac{d\hat{z}_i}{dt} - g_i(\hat{z}_1, \ldots, \hat{z}_7; \theta) \right\|^2$$

where $\frac{d\hat{z}_i}{dt}$ is the analytic time derivative computed through the FiLM pathway (described in the architecture document), and $g_i(\hat{z})$ is the ODE right-hand side evaluated at the predicted states. The total physics loss averages these residuals across all six non-neural equations (the ODE for $x$ is not included because $x$ is the free driving input, not a variable with its own ODE), all three layers, and all collocation points.

**Numerical stabilisation.**

The ODE right-hand sides involve divisions and fractional exponents that can produce numerical instabilities when the predicted states take extreme values, particularly early in training. Before evaluating the right-hand sides, the predicted states are sanitised: NaN and infinity values are replaced with sensible defaults, and the positive-definite variables $f$, $v$, and $q$ are clamped to a minimum of $0.1$ to prevent division by near-zero values or undefined fractional powers. This sanitisation is applied only for the physics loss computation, not to the states returned by the network.

**The burn-in period.**

The ODE residuals are evaluated only after a short burn-in of initial time steps. This is because the network's predictions near $t = 0$ are unreliable: the temporal encoder has seen very little past context, and the network has not yet established a consistent estimate of the initial state. Including these early time points in the physics loss would introduce noisy gradients. The burn-in discards a fixed number of initial collocation time indices before computing the residual.

**Failure mode without this term.** Without the physics loss, the network can produce latent trajectories that reproduce BOLD perfectly but are physically impossible: $v$ and $q$ could oscillate arbitrarily, $s$ could be uncoupled from $x$, and the inversion would have no physical interpretation. The physics loss is what gives the recovered latents their meaning as hemodynamic quantities.

---

## 6. The Supervision Loss

**_Note: This is purely a debugging tool, and not a design goal._**

The supervision loss is not a principled component of the final training objective. It is present during the current development phase as a diagnostic: if the model cannot recover the latent states even when given direct supervision at the source voxel, then the failure lies in the architecture or the physics loss, not in the difficulty of the unsupervised problem. Conversely, if the model succeeds under supervision but fails without it, the unsupervised signal (data loss plus physics loss) is insufficient on its own and needs to be strengthened. The supervision loss is therefore a stepping stone: it is used now to establish that the model is *capable* of recovering the latents, before the harder question of whether it can do so without any ground-truth guidance is addressed.

The long-term objective is a model that recovers the latent states from BOLD alone, with no access to ground-truth latents at any point during training. The supervision loss will be removed once it has served its diagnostic purpose.

**What it computes.**

The supervision loss applies MSE between the predicted and ground-truth latent states at the source voxel, across all layers and the full time series. It is applied to $s$, $f$, $v$, $q$, $v^*$, and $q^*$. Note that $x$ is not directly supervised: the neural activity is the quantity the model is ultimately trying to recover, and supervising it directly would bypass the inversion entirely rather than test it.

The supervision is restricted to the source voxel rather than the full spatial grid. This reflects the fact that only one voxel per training sample has a known active neural source; supervising background voxels would provide misleading signal since those voxels should be at or near resting state.

**What success under supervision would tell us.**

If the model recovers $s$, $f$, $v$, and $q$ accurately at the source voxel under supervision, and the physics loss is also well-satisfied, then the architecture is expressive enough to represent the correct solution. The remaining question is whether the data and physics losses alone, without ground-truth guidance, can steer the network to that solution during unsupervised training on real data. That is the harder and more important question, and it is the one the current development phase is working toward.

---

## 7. Loss Scheduling and the Warm-Up Strategy

**Why scheduling is necessary.**

The three loss terms do not play equally well together at all stages of training. The most dangerous interaction is between the data loss and the physics loss early in training. At initialisation, the network's predictions are essentially random: the latents have no meaningful structure, and the BOLD reconstruction is poor. If the physics loss is applied at full strength from the beginning, it will dominate the gradient signal and drive the network toward the ODE resting-state fixed point, where all variables sit at their equilibrium values. This satisfies the physics loss trivially but destroys the data loss signal. Once the network has converged to the fixed point, escaping it is very difficult.

The solution is to delay and ramp the physics loss. During the early phase of training, only the data loss is active. The network is free to find any solution that roughly reproduces the BOLD signal. Once the data loss has established a reasonable initialisation, the physics loss is introduced gradually, so that the network can adapt its predictions toward physical consistency without losing the BOLD signal entirely.

**How scheduling is implemented.**

Each scheduled loss term has two parameters: a *delay* and a *warm-up duration*. During the delay period, the effective weight of that term is exactly zero. After the delay, the weight is linearly ramped from zero to its target value over the warm-up duration. Formally:

$$\lambda(t) = \begin{cases} 0 & t < t_{\text{delay}} \\ \lambda_{\text{target}} \cdot \min\left(1,\, \frac{t - t_{\text{delay}}}{t_{\text{warmup}}}\right) & t \geq t_{\text{delay}} \end{cases}$$

where $t$ is the global training step. The scheduling is applied to both the physics loss and the supervision loss. The data loss is active from the beginning and is not scheduled.

An important implementation detail: the time derivative computation in the decoder, which is needed for the physics loss, is expensive. If the physics loss weight is zero, computing the derivatives would waste compute. The code therefore checks whether the effective physics weight is positive before requesting gradients from the forward pass, avoiding this cost during the delay period.

---

## 8. The Validation Procedure

At validation time, the model is evaluated on held-out synthetic data with known ground-truth latents. This is the closest available proxy to an oracle evaluation: we can directly compare the predicted latent trajectories against the true ones without any ambiguity.

Validation runs the full forward pass including time derivatives, regardless of the physics loss schedule, since the schedule is a training device and validation should always reflect the full model capability. The same losses as in training are computed and logged, but the primary diagnostic is the visual comparison of predicted and ground-truth trajectories at the source voxel.

Two sets of plots are generated at the end of each validation epoch. The first compares predicted and true BOLD signals alongside predicted and true neural activity $x$ across all three layers, for a random subset of validation samples. The second compares the full set of predicted and true latent trajectories ($s$, $f$, $v$, $q$, $v^*$, $q^*$) in the same format. These plots are logged to Weights and Biases and serve as the primary qualitative indicator of training progress.

---

## 9. The Training Objective as a Whole

It is worth pausing to appreciate the structure of what has been described. The network is being asked to invert a nonlinear dynamical system from its outputs alone, without direct observation of its internal states. The two principled loss terms define a constrained optimisation problem: find latent trajectories that (a) reproduce the observed BOLD when passed through the forward model, and (b) satisfy the governing ODEs. The supervision loss is present only temporarily as a development scaffold.

Each of the two principled constraints alone is insufficient. Constraint (a) alone is underdetermined: many latent trajectories can reproduce a given BOLD trace. Constraint (b) alone has a trivial solution at the resting-state fixed point, where all variables sit at equilibrium and nothing is recovered. Together, they triangulate a solution that is simultaneously data-consistent and physically plausible.

The supervision loss currently provides a third constraint that anchors the solution at the source voxel. Its purpose is diagnostic: establishing that the model architecture and physics loss are expressive and well-calibrated enough to recover the correct latents when given some direct guidance. Once that is confirmed, the supervision will be removed and the model will be required to recover the latents from constraints (a) and (b) alone. That is the true objective of the project.

The warm-up scheduling exists because the interaction between the data and physics losses is fragile early in training. The ordering matters: the data loss must first establish a non-trivial solution, then the physics loss can refine it toward physical consistency. Violating this ordering by introducing the physics loss too early tends to collapse training to the resting-state fixed point, from which recovery is very difficult. Understanding this ordering is as important as understanding the individual loss terms themselves.
