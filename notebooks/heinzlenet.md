# HeinzleNet: A Structured Spatio-Temporal Architecture for Layered Physiological Dynamics


## 1. Introduction and Motivation

Before describing the architecture, it is worth asking: why not use a standard neural network? If we have a spatio-temporal input and want to predict a set of physiological states, we could in principle feed the data into a generic sequence model or a large convolutional network and let it learn whatever it wants.

The answer is that we have prior knowledge about the structure of the problem, and ignoring that prior knowledge is wasteful: it forces the network to learn things we already know, increases the number of free parameters, and makes it harder to diagnose failures. The design of HeinzleNet is therefore not arbitrary. Every architectural choice encodes a specific assumption about the data or the underlying physiology. Part of understanding this architecture is understanding *why* each component is where it is.

The problem we are solving is as follows. We observe a BOLD fMRI signal measured across a spatial grid of voxels and across multiple cortical layers and time. We want to recover the latent hemodynamic and neural state variables that generated those observations. Specifically, these are the seven variables of the Heinzle laminar hemodynamic model: neural activity $x$, vasodilatory signal $s$, blood inflow $f$, blood volume $v$ and $v^*$, and deoxyhaemoglobin content $q$ and $q^*$, each estimated independently per cortical layer. The network has no access to these latent variables at inference time on real data, so the architecture must impose enough structure that a physically plausible inversion becomes learnable from BOLD alone. During the current training regime on synthetic data, ground-truth latents are in fact used as a substantial supervision signal (see `training.md`); that dependency does not exist for real fMRI, which is exactly why the architecture's structure matters.

---

## 2. The Input Tensor and What Its Axes Mean

The network receives a five-dimensional tensor of observed BOLD signals:

$$\mathbf{x} \in \mathbb{R}^{B \times L \times T \times H \times W}$$

Each axis has a distinct physical interpretation.

$B$ is the batch size, i.e. the number of independent examples processed simultaneously. This is purely a computational convenience and carries no physical meaning.

$L$ is the number of cortical layers. Every block that depends on layer count (`MaskedLayerMixing`, `SpatioTemporalDecoder`) takes `L` as a constructor argument, and `train_mich.py` overwrites `cfg.model.L` at the start of every run from the metadata stored in the target simulation's HDF5 file, before the model is built. A three-layer scenario with vascular drainage (deep, middle, superficial) gives $L=3$; a single-layer scenario gives $L=1$, with no drainage terms and 5 rather than 7 Heinzle signals decoded (see §8). $L=3$ is a common case.

$T$ is the number of time steps (for our data, around 300 timepoints at roughly 1-second resolution).

$H$ and $W$ are the height and width of the spatial grid of voxels. Like $L$, this is a config value (`model.grid_size`) rather than a fixed architectural constant; the network is fully convolutional over $H, W$ and imposes no constraint on grid size. The configs currently use a $10\times10$ grid.

A key property of this tensor is that the axes are *semantically independent*: the spatial structure of the data is captured by $H$ and $W$, the temporal structure by $T$, and the laminar structure by $L$. This independence is what the architecture will exploit, by processing each axis with components designed for it, rather than collapsing everything into one undifferentiated representation.

---

## 3. A Primer on Convolution

Since this architecture relies heavily on convolutions, it is worth reviewing the key concept before proceeding.

A convolution is an operation that slides a small *filter* (also called a *kernel*) across an input, computing a weighted sum at each position. The weights of the filter are learned. The key properties relevant here are:

**Local connectivity.** The output at each position depends only on a small neighborhood of the input. This encodes the assumption that nearby positions share statistical structure, which is a reasonable prior for both spatial images and temporal sequences.

**Weight sharing.** The same filter weights are applied at every position. This dramatically reduces the number of parameters compared to a fully connected network, and encodes the assumption that the statistical structure is *translation-invariant*: the same patterns are meaningful regardless of where they appear.

**Channels.** In practice, we apply not one filter but many, each producing a separate *feature map*. The collection of all feature maps at a given layer constitutes the *channels* of that layer's representation.

A **1×1 convolution** is a special case where the filter has spatial size 1×1. It therefore does no spatial mixing at all. Instead, it computes a linear combination of the channels at each spatial position independently. This is useful when we want to change the number of channels or mix information across channels without affecting spatial structure.

A **depthwise separable convolution** factorises a standard convolution into two steps: a *depthwise* step that applies a separate filter independently to each input channel (no channel mixing), followed by a *pointwise* 1×1 convolution that mixes channels. This is computationally cheaper than a full convolution and can be easier to train when channel mixing and spatial filtering serve distinct roles.

A **dilated convolution** is commonly used for temporal sequence modeling. *Dilated* means the filter skips positions by a fixed factor, so a filter of kernel size 3 with dilation $d$ covers positions $\{t-d, t, t+d\}$ (or $\{t, t-d, t-2d\}$ if the convolution is made *causal*, meaning the filter extends only backwards in time so the output at time $t$ depends only on times $t' \leq t$). Stacking layers with exponentially increasing dilations $d = 1, 2, 4, 8, \ldots$ gives a receptive field that grows exponentially with depth, allowing the network to capture long-range dependencies without requiring very large kernels. §7 describes which of these two padding choices HeinzleNet's temporal encoder currently uses.

---

## 4. The Overall Pipeline

The computation in HeinzleNet proceeds through four sequential stages:

$$\text{Input} \;\xrightarrow{\;\text{Layer Mixing}\;}\; \text{Spatial Encoding} \;\xrightarrow{\;}\; \text{Temporal Mixing} \;\xrightarrow{\;}\; \text{Structured Decoding} \;\xrightarrow{\;}\; \text{Output States}$$

Each stage is responsible for one aspect of the problem. An important implementation detail runs through all four stages: rather than carrying the layer dimension $L$ as a separate axis throughout, it is *folded into the batch dimension*. This means that instead of treating the layers as separate inputs to be processed in parallel by separate streams, the network treats each (batch element, layer) pair as an independent sample. It is a computational convention that allows standard convolutional modules to be reused across layers without modification, while the learned parameters remain shared across layers unless explicitly made layer-specific (as in the decoder).

---

## 5. Stage 1: Layer Mixing

**What problem does this solve?**

Cortical layers are dependent on one another. The hemodynamic response in the superficial layer is influenced by the deep layer, both through direct vascular coupling and through shared neural drive. Before the network processes spatial or temporal structure, it should first allow information to flow between layers at each spatial location and time point.

**How it works.**

The input has shape $[B, L, T, H, W]$. We begin by collapsing the batch and time dimensions together:

$$[B, L, T, H, W] \;\rightarrow\; [B \cdot T,\; L,\; H,\; W]$$

This means each $(b, t)$ slice becomes an independent image with $L$ channels. We then apply a 1×1 convolution with $L$ input and $L$ output channels. Because the kernel is 1×1, it touches only the channel (layer) dimension at each spatial location, so no spatial mixing occurs. Because time has been collapsed into the batch, no temporal mixing occurs either. This is a pure layer-to-layer mixing operation.

**The mask.** The weight matrix of this convolution is multiplied element-wise by a binary mask $M \in \{0,1\}^{L \times L}$:

$$W_{\text{eff}} = W \odot M$$

The mask enforces the assumption that cortical layers interact locally: each layer $\ell$ may receive contributions only from itself and its immediate predecessor in the laminar hierarchy. This is not an arbitrary sparsity constraint; it reflects our prior knowledge about how information propagates across cortical depth.

**Channel expansion.** Following the masked mixing, a second 1×1 convolution expands each layer's representation into a $C = 32$ dimensional feature space. Crucially, this expansion is applied *per layer independently*: the $L$ layers are treated as separate single-channel inputs (`nn.Conv2d(1, C, kernel_size=1)`), so the expansion lifts each individual layer's scalar value into $C$ channels, rather than projecting across all $L$ layers jointly. The shape at this point is $[B \cdot T \cdot L, C, H, W]$, restored to $[B, T, L, C, H, W]$ before passing onward. The $L$ layers are now carried forward as separate elements in the batch dimension.

---

## 6. Stage 2: Spatial Encoding

**What problem does this solve?**

Each voxel in the BOLD signal is influenced not just by its own underlying neural activity, but also by the hemodynamic point spread function. Nearby neural activity blurs into a given voxel's signal. Furthermore, the spatial context of a voxel (its neighborhood structure, the pattern of activation in surrounding voxels) carries information relevant to inverting the signal. The spatial encoder extracts these local spatial features.

**How it works.**

The input at this stage has shape $[B \cdot T \cdot L, C, H, W]$, with time and layer both folded into the batch. This is the standard format for a 2D convolutional network: a batch of images, each with $C$ channels and spatial extent $H \times W$.

Two layers of depthwise separable convolution are applied in sequence. Each layer consists of:

1. A depthwise convolution with a 3×3 kernel, applying a separate spatial filter to each channel independently.
2. A pointwise 1×1 convolution that mixes channels.
3. Group normalisation, which normalises each subset of channels to have zero mean and unit variance, stabilising training.
4. A SiLU nonlinearity (a smooth gating function that tends to train well in deep networks).

No temporal or layer interactions are introduced here. The output is a richer spatial representation at each time point and each layer, with shape $[B \cdot T \cdot L, C', H, W]$ where $C' = 64$.

---

## 7. Stage 3: Temporal Mixing

**What problem does this solve?**

The hemodynamic response function (HRF) has a temporal extent of roughly 20–30 seconds. Neural activity at time $t$ influences the BOLD signal over a window extending many seconds into the future. To invert this process, the network needs access to temporal context spanning the full duration of the HRF. The temporal mixing stage provides this.

**How it works.**

The tensor is reorganised so that each spatial location and each layer defines a separate time series. Specifically:

$$[B, T, L, C', H, W] \;\xrightarrow{\text{permute}}\; [B, L, H, W, C', T] \;\xrightarrow{\text{reshape}}\; [B \cdot L \cdot H \cdot W,\; C',\; T]$$

This gives a batch of 1D sequences of length $T$, each with $C'$ channels, one per (batch, layer, spatial position). A temporal convolutional network (TCN) is then applied to all of these sequences simultaneously.

The TCN consists of six layers of dilated depthwise separable convolutions with dilations $d = 1, 2, 4, 8, 16, 32$, each with a kernel size of 3. The padding used to preserve sequence length is split evenly between the two sides of the input at every layer, so the network is **not causal**: the output at time $t$ depends on both past and future input samples. Concretely, for kernel size $k=3$, each layer's tap positions are $\{t-d, t, t+d\}$ rather than the backward-only $\{t-2d, t-d, t\}$ a causal layer would use. Stacking all six dilations gives a total receptive field of $1 + (k-1)\sum_i d_i = 1 + 2 \cdot 63 = 127$ time steps, extending roughly 63 steps into the past and 63 steps into the future of each output position. This is fine for training and offline evaluation, where the full time series is available at once, but it does mean the temporal encoder's features at time $t$ are not solely a function of the past. (An earlier version of this stage did use causal, left-only padding; it was changed to symmetric padding, which is the current behavior described here.)

Each TCN layer includes a residual connection: the input is added to the output, so the network learns *corrections* to the input representation rather than building it from scratch at each layer. This is important for training stability in deep networks.

After the TCN, the tensor is restored to $[B, T, L, C', H, W]$ via the inverse reshape and permutation.

---

## 8. Stage 4: Structured Decoding

**What problem does this solve?**

After spatial and temporal encoding, we have a rich feature representation at each (batch, time, layer, spatial location). We now need to map these features into the Heinzle state variables per layer, in a way that respects the known structure of those variables: their physical units, their positivity constraints, and their dependence on time, layer identity, and signal identity.

The signal set is not fixed: it is `HEINZLE_SIGNALS` (`x, s, f, v, q, v*, q*`, 7 signals) for multi-layer datasets with a draining-vein compartment, or `HEINZLE_SIGNALS_SINGLE` (`x, s, f, v, q`, 5 signals) for single-layer datasets. `train_mich.py` chooses between the two and sets `cfg.model.heinzle_net.spatial_decoder_config.signals` accordingly, based on the layer count read from the simulation metadata (§2). The rest of this section describes the mechanism generically; substitute $N$ for the signal count (5 or 7) where relevant.

**The challenge: time-dependent, signal-specific decoding.**

A naive approach would be to apply a single linear projection from features to outputs at each time point. But we want the decoder to be aware of three kinds of metadata: *which time point* we are at (because the same feature vector should map to different outputs at different phases of the hemodynamic cycle), *which layer* we are in (because the hemodynamic parameters differ across layers), and *which signal* we are predicting (because each variable has different physical meaning and constraints).

FiLM, or Feature-wise Linear Modulation, provides an elegant solution to this. The idea is simple: given some conditioning information (here, time, layer, and signal identity), we compute a scale $\gamma$ and a shift $\beta$, and apply them to a feature vector:

$$\tilde{z} = \gamma \odot z + \beta$$

This is an affine transformation, with the transformation itself being a function of the conditioning information. The network therefore learns not a single fixed mapping from features to outputs, but a *family* of mappings parameterised by time, layer, and signal. The feature vector FiLM modulates is not the full $c_{\text{dec}}$-dimensional decoder representation ($c_{\text{dec}} = 64$ by default): a shared $1\times1$ convolution first projects it down to a smaller $c_{\text{film}}$-dimensional bottleneck (32 by default), and FiLM operates in that bottleneck space. This keeps the FiLM parameter budget from scaling with the full decoder width while still giving each signal its own dedicated temporal dynamics.

**Time embedding.** Time is embedded using Fourier features: for a physical time $t$ in seconds, we compute $[\sin(2\pi f_k t), \cos(2\pi f_k t)]$ for a set of frequencies $f_k$. This gives the decoder access to a continuous representation of time that can capture periodic and transient structure. Critically, because these are analytic functions, we can later differentiate the decoder output with respect to time exactly, which is needed for the physics loss. The frequency range and count are config values (`model.num_freqs`, `model.min_freq`, `model.max_freq`), not fixed constants; the default config currently uses 24 log-spaced frequencies from 1 Hz to 48 Hz, chosen to cover both the single dominant low-frequency mode and the near-Nyquist content in $x$ and $s$ (the sampling rate gives a Nyquist frequency of about 49.5 Hz). A smaller/narrower variant (10 frequencies, 5 to 20 Hz) exists as a non-default model config.

**Layer and signal embeddings.** Each layer and each signal is assigned a learned embedding vector. These embeddings are concatenated to the Fourier time embedding and passed through a small multilayer perceptron to produce the FiLM parameters $(\gamma, \beta)$.

**Per-(signal, layer) output heads.** After FiLM modulation, each of the $N \times L$ (signal, layer) combinations has its own independent linear projection head: a 1×1 convolution that maps from the $c_{\text{film}}$-dimensional bottleneck to a scalar prediction at each spatial location. This is not a shared head. The weights of the deep-layer $q$ head are entirely independent of the superficial-layer $q$ head, and both are independent of the $v$ heads. This gives the decoder the freedom to learn genuinely different mappings for each variable at each layer.

**Output shape.** The final prediction tensor has shape:

$$\hat{z} \in \mathbb{R}^{B \times N \times L \times T \times H \times W}$$

where the second dimension indexes the $N$ Heinzle variables (5 or 7, see above) in fixed order.

---

## 9. Output Constraints

Not all Heinzle variables are physically unconstrained. Blood inflow $f$, blood volume $v$, and deoxyhaemoglobin $q$ are constrained to be strictly positive via a softplus activation:

$$\text{softplus}(u) = \log(1 + e^u)$$

which is a smooth, everywhere-positive function that saturates to a linear response for large positive inputs and decays smoothly toward zero for large negative ones. The vasodilatory signal $s$, venous volume $v^*$, and venous deoxyhaemoglobin $q^*$ are left unconstrained (identity activation). Of these, $s$ in particular can be negative, representing active vasoconstriction.

Neural activity $x$ is also identity-activated, not softplus. It was constrained to be positive in an earlier version of the architecture; that constraint has since been removed, because the neural-baseline generator can inject signed offsets into inter-pulse rest periods (a "random" baseline mode) and nothing downstream requires $x \geq 0$. `_sanitise_states` (used only for the physics loss's numerical stabilisation, §5 in `training.md`) clamps $x$ to $[-1000, 1000]$, the same branch as $s$, $v^*$, and $q^*$, not the non-negative branch used for $f$, $v$, $q$. The network can therefore represent net inhibitory states as negative $x$ values.

---

## 10. Time Derivatives

The physics loss requires access to $\frac{d\hat{z}}{dt}$, the time derivative of each predicted state variable, because the Heinzle ODEs express relationships between state variables and their derivatives. Computing this accurately is important for the physics loss to be meaningful.

Let $u$ denote the spatial features entering the decoder (the output of temporal mixing, before the FiLM projection), and let $z_{\text{pre}} = u_{\text{film}} \cdot \gamma(t) + \beta(t)$ be the pre-activation decoder output, where $u_{\text{film}}$ is $u$ projected into the FiLM bottleneck (§8). An earlier version of this computation treated $u$ (and hence $u_{\text{film}}$) as constant with respect to $t$, differentiating only through the FiLM conditioning pathway: $\gamma(t)$ and $\beta(t)$. That is exact for the FiLM branch in isolation, but it misses the fact that $u$ itself carries time-dependence, since it is the output of `temporal_mixing`'s TCN (§7), which is where the signal's own dynamics actually live. An ablation confirmed this mattered in practice: replacing $u$ with its temporal mean collapsed the resulting derivative gap but broke the model's ability to fit the data, showing the dynamics really do route through $u$.

The current implementation computes the full product rule:

$$\frac{dz_{\text{pre}}}{dt} = \frac{du_{\text{film}}}{dt} \cdot \gamma(t) + u_{\text{film}} \cdot \frac{d\gamma}{dt} + \frac{d\beta}{dt}$$

$\frac{d\gamma}{dt}$ and $\frac{d\beta}{dt}$ are computed exactly, by differentiating through the Fourier time embedding and the FiLM MLP (the Fourier embedding is an analytic function of $t$, so its derivative is available in closed form). $\frac{du}{dt}$ has no such closed form, since $u$ only exists at the $T$ recorded time samples with no continuous parameterisation in $t$; it is instead estimated by a central finite difference (`torch.gradient`) over the sampled time grid. The bottleneck projection from $u$ to $u_{\text{film}}$ is a per-timestep-shared, bias-free linear map, so it commutes exactly with the time derivative: projecting $du/dt$ gives the same result as differentiating the projected features directly.

Two further implementation details are worth noting. First, the $d\gamma/dt$, $d\beta/dt$ computation is split into $L$ separate `vmap(jacrev(...))` calls, one per cortical layer, rather than one joint call. This is necessary to avoid tracing through `nn.Embedding`, which is not differentiable with respect to integer indices. The layer and signal embeddings are therefore looked up once and held constant, while `jacrev` differentiates only through the Fourier embedding and the FiLM MLP. Second, when applying the output projection heads to compute $\frac{d\hat{z}_{\text{pre}}}{dt}$, the bias terms are explicitly dropped (set to `None`). This is correct because the bias is a constant and its derivative with respect to time is zero, so only the weight matrix contributes to the derivative.

The chain rule then gives the final time derivative of the post-activation output:

$$\frac{d}{dt}\hat{z} = \sigma'(\hat{z}_{\text{pre}}) \odot \frac{d\hat{z}_{\text{pre}}}{dt}$$

where $\sigma'$ is the pointwise derivative of the activation function for each channel.

This entire branch, from the finite-difference $du/dt$ through the chain rule above, runs in float32 regardless of the training precision elsewhere, to keep a single clean precision boundary around the `vmap`/`jacrev` composition rather than introducing a second fp32/bf16 seam next to it.

---

## 11. The Architecture as a Whole: What Each Component Contributes

It is worth stepping back and stating plainly what each stage contributes to the overall inversion.

The **layer mixing** stage allows the model to account for the fact that the signal in a given layer is partly driven by hemodynamic coupling from neighbouring layers. Without this, the encoder would process each layer in complete isolation.

The **spatial encoder** extracts local spatial features that encode neighbourhood context. This is important because the point spread function blurs activity across voxels, and recovering clean per-voxel estimates requires knowing something about the surrounding spatial pattern.

The **temporal encoder** provides temporal context spanning the full HRF. Without this, the decoder would have to invert a long-memory process from a single time point, which is not possible.

The **decoder** imposes physiologically meaningful structure on the output: time-awareness via Fourier embeddings, layer-specificity via layer embeddings, signal-specificity via per-head projections, and physical constraints via activation functions.
