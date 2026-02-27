# MICH Forward Pass

### Signal Definition

Everything starts with a spatio-temporal BOLD signal:

$$s \in \mathbb{R}^{B \times L \times T \times H \times W}$$

$B$ is the batch size, $L = 3$ is the number of cortical layers, $T = 3000$ is the number of timepoints, and $H = W = 32$ are the spatial dimensions. So each sample in the batch is a 3-layer stack of 32×32 spatial grids evolving over 3000 timepoints.

This signal passes sequentially through four modules inside `HeinzleNet`: `MaskedLayerMixing`, `SpatialEncoder`, `TemporalMixingEncoder`, and `SpatioTemporalDecoder`. Each module is designed to handle one specific aspect of the signal's structure : layer, space, and time : in isolation before they are brought together in the decoder. This separation is intentional and worth keeping in mind as you read through.



#### 1. `MaskedLayerMixing`

The first thing we want to do is let the network learn how the three cortical layers relate to each other. In the real cortex, layers are not independent : superficial layers receive input from deeper ones, and there is a known draining hierarchy in the BOLD signal. This module gives the network a structured way to learn those relationships, while being explicit about what it is *not* allowed to do: mix spatial locations or mix timepoints.

**Input:** $s \in \mathbb{R}^{B \times L \times T \times H \times W}$

**Output:** $y \in \mathbb{R}^{B \times T \times C \times H \times W}$



**Step 1 : Collapse batch and time.**
To apply a 2D convolution that only touches the layer dimension, we first fold the time axis into the batch axis. This means each $(b, t)$ combination is treated as a separate, independent image:

$$[B, L, T, H, W] \;\rightarrow\; [B{\cdot}T, L, H, W]$$

Now we have a large collection of $L$-channel spatial frames, and any operation we apply will be blind to which batch item or timepoint a frame came from.

**Step 2 : Masked 1×1 layer mixing** ($L \rightarrow L$).
A $1{\times}1$ convolution is applied across the layer channels. Because the kernel is $1{\times}1$, it cannot look at neighbouring pixels : it only combines the three layer values at each spatial location independently. A fixed binary mask $M \in \{0,1\}^{L \times L}$ zeros out certain weights to enforce a specific coupling structure (for example, to respect the cortical draining direction):

$$W_{\text{eff}} = W \odot M$$

The output at each spatial location $(h, w)$ for each collapsed frame $(b, t)$ is:

$$z_{(b,t),\,\ell,\,h,w} = \sum_{k=1}^{L} W_{\text{eff}}[\ell, k]\; s_{(b,t),k,h,w} + b[\ell]$$

The shape is unchanged: $[B{\cdot}T, L, H, W]$.

**Step 3 : Channel expansion** ($L \rightarrow C$).
Three layer channels is a very limited feature space. A second $1{\times}1$ convolution expands this to $C$ channels, giving the network more representational capacity to work with downstream:

$$[B{\cdot}T, L, H, W] \;\rightarrow\; [B{\cdot}T, C, H, W]$$

**Step 4 : Restore axes.**
Finally we unfold the batch and time dimensions back to where they belong:

$$[B{\cdot}T, C, H, W] \;\rightarrow\; [B, T, C, H, W]$$



By the end of this module, the signal has gone from a 3-layer laminar representation to a $C$-channel feature map, with layer relationships encoded in a controlled, interpretable way. No pixels have talked to their neighbours, and no timepoint has influenced another.



#### 2. `SpatialEncoder`

Now that laminar structure has been handled, we want to extract spatial features : edges, gradients, local patterns in the activation maps. This is where standard convolutional processing happens. The key design choice here is the same as before. Time is kept completely out of the picture. We collapse it into the batch dimension again so that the spatial convolutions cannot inadvertently mix information across timepoints.

**Input:** $x_\text{mix} \in \mathbb{R}^{B \times T \times C \times H \times W}$

**Output:** $x_\text{enc} \in \mathbb{R}^{B \times T \times C' \times H' \times W'}$

where $C'$ is the number of output channels and $H', W'$ are the spatial dimensions after any downsampling.



**Step 1 : Permute and collapse time.**
We merge the time axis and the batch axis:

$$[B, T, C, H, W] \;\rightarrow\; [B{\cdot}T, C, H, W]$$

We now have a large stack of independent 2D feature maps. The spatial convolutions that follow have no way of knowing which timepoint any given frame belongs to.

**Step 2 : Depthwise-separable convolution blocks (×N).**
A stack of $N$ convolutional blocks is applied. We use depthwise-separable convolutions rather than standard convolutions because they are much more parameter-efficient, which matters when your spatial grid is only 32×32 and you want to avoid overfitting.

Each block does four things in sequence. First, a depthwise convolution applies a separate $3{\times}3$ kernel $k^{(c)}$ to each channel independently:

$$x^{(dw)}_{n,c} = k^{(c)} * x_{n,c}$$

This is purely about spatial structure : each channel looks at its local neighbourhood but does not interact with the others. If a stride greater than 1 is used here, the spatial dimensions are downsampled.

Second, a pointwise ($1{\times}1$) convolution then mixes information across channels, projecting from $C_\text{in}$ to $C_\text{out}$:

$$x^{(pw)}_{n,:,h,w} = W_{pw}\, x^{(dw)}_{n,:,h,w}$$

Third, Group Normalisation is applied to stabilise the activations:

$$\hat{x} = \frac{x^{(pw)} - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}}$$

Unlike Batch Normalisation, GroupNorm does not depend on batch statistics, which makes it a better fit here since temporal correlation means the effective batch diversity is lower than it looks.

Fourth, a SiLU nonlinearity is applied:

$$\phi(\hat{x}) = \hat{x} \cdot \sigma(\hat{x})$$

This introduces the nonlinearity needed to learn hierarchical spatial features across the stack of blocks.

**Step 3 : Restore axes.**
After all $N$ blocks, we separate batch and time again:

$$[B{\cdot}T, C', H', W'] \;\rightarrow\; [B, T, C', H', W']$$



At this point the signal is a sequence of spatial feature maps : one per timepoint : with richer spatial structure than the raw input, but still no temporal relationships encoded. That is the job of the next module.

#### 3. `TemporalMixingEncoder`

So far the network has learned how layers relate to each other and what spatial structure looks like at each timepoint, but it has been deliberately blind to how the signal evolves over time. This module addresses that. It processes each spatial location as an independent time series, learning temporal dynamics through a stack of dilated 1D convolutions.

**Input:** $x_\text{enc} \in \mathbb{R}^{B \times T \times C' \times H \times W}$

**Output:** $x_\text{temp} \in \mathbb{R}^{B \times T \times C' \times H \times W}$

The shape is unchanged : this module enriches the temporal structure of the representation without altering its dimensions.



**Step 1 : Collapse spatial locations into the batch dimension.**
We want to apply 1D convolutions along the time axis. To do this independently at every spatial location, we fold $H$ and $W$ into the batch dimension and rearrange so that time is the last axis, as Conv1d expects:

$$[B, T, C', H, W] \;\rightarrow\; [B, H, W, C', T] \;\rightarrow\; [B{\cdot}H{\cdot}W,\; C',\; T]$$

Each of the $B \cdot H \cdot W$ entries is now an independent $C'$-channel time series of length $T$. Crucially, no spatial mixing happens here : a pixel in the top-left corner of the grid cannot influence one in the bottom-right.

**Step 2 : Stack of dilated depthwise TCN layers.**
A sequence of $N$ temporal convolutional blocks is applied. Each block is a `TemporalDepthWiseTCNLayer` with an exponentially increasing dilation following the pattern $1, 2, 4, 8, \ldots, 2^{N-1}$.

Why dilation? With a kernel size of 3 and no dilation, each output timepoint can only see its immediate neighbours. Doubling the dilation at each layer exponentially expands the receptive field : after $N$ layers, the network can relate timepoints that are $2^N$ steps apart. For $T = 3000$ this matters: hemodynamic responses are slow and the relevant temporal context spans hundreds of timepoints.

Each block applies the following operations. First, a depthwise 1D convolution with dilation $d$ filters each channel independently along time:

$$x^{(dw)}_{n,c} = k^{(c)} *_d\, x_{n,c}$$

where $*_d$ denotes convolution with dilation $d$. The padding is set to $(k-1) \cdot d \;/\; 2$ to preserve the sequence length, so $T$ does not change.

Second, a pointwise ($1{\times}1$) convolution mixes information across channels without touching the time axis:

$$x^{(pw)}_{n,:,t} = W_{pw}\, x^{(dw)}_{n,:,t}$$

Third, Group Normalisation stabilises the activations within each channel group. Fourth, a SiLU nonlinearity is applied.

Finally, a **residual connection** adds the block's input back to its output:

$$x_\text{out} = \phi\!\left(\text{Norm}\!\left(W_{pw} * \left(k^{(\cdot)} *_d x\right)\right)\right) + x$$

This is what makes stacking many dilated layers practical. Without it, gradients vanish quickly through a deep temporal stack. With it, each block only needs to learn a correction to its input rather than a full transformation, which is much easier to optimise. For a refresher on how and why skip connections help, check out this [link](https://medium.com/@sandushiw98/understanding-resnet-50-solving-the-vanishing-gradient-problem-with-skip-connections-5591fcb7ff74).

**Step 3 : Restore axes.**
After all $N$ blocks, we reshape back to the original layout:

$$[B{\cdot}H{\cdot}W, C', T] \;\rightarrow\; [B, H, W, C', T] \;\rightarrow\; [B, T, C', H, W]$$


After this module, each spatial location holds a temporally rich feature vector that has seen context spanning thousands of timepoints through the exponentially growing receptive field. The representation now carries laminar structure, spatial structure, and temporal dynamics, and is ready to be decoded back into the neural signal space.


#### 4. `SpatioTemporalDecoder`

The decoder takes the enriched feature representation and maps it back to something physically meaningful: an estimate of the neural and hemodynamic state variables at every timepoint, layer, and spatial location. It also introduces something new : the timepoint index $t$ is passed in explicitly as a continuous variable and used to condition the decoding through a mechanism called FiLM. This is also where optional time derivatives of the output are computed, which are used downstream in the physics-informed loss.

**Input:** $x_\text{temp} \in \mathbb{R}^{B \times T \times C' \times H \times W}$, and $t \in \mathbb{R}^{B \times T}$

**Output:** $\hat{z} \in \mathbb{R}^{B \times 7 \times L \times T \times H \times W}$

The 7 output channels correspond to the state variables of the hemodynamic model (one set per cortical layer $L$).



##### Time Conditioning via FiLM
---

Before the spatial decoding happens, the decoder computes a time-dependent scaling and shift for every feature map. This is the FiLM (Feature-wise Linear Modulation) mechanism and it is worth understanding before looking at the full forward pass.

**`FourierTimeEmbedding`** takes the raw timepoint indices $t \in \mathbb{R}^{B \times T}$ and maps them into a higher-dimensional embedding using sinusoidal features at logarithmically spaced frequencies:

$$\omega_f = 2\pi \cdot t \cdot f_i, \quad f_i \in \text{logspace}(1, f_\text{max}, F)$$

$$e_t = \left[\sin(\omega_1),\, \cos(\omega_1),\, \ldots,\, \sin(\omega_F),\, \cos(\omega_F)\right] \in \mathbb{R}^{B \times T \times 2F}$$

Using both sine and cosine at each frequency gives the embedding sensitivity to both phase and magnitude of temporal variation. The logarithmic spacing ensures the embedding captures both fast and slow temporal patterns.

**`TimeFiLM`** then takes this embedding and produces a per-channel scale $\gamma$ and shift $\beta$ through a small MLP:

$$\gamma, \beta = \text{chunk}\!\left(W_2\,\phi\!\left(W_1\, e_t\right)\right) \in \mathbb{R}^{B \times T \times C_\text{dec}}$$

These will be used to modulate the spatial features channel-wise, allowing the decoding to be aware of *when* in time each frame sits.



**Step 1 : Spatial feature extraction.**
The temporal features are first projected into the decoder's channel space using a depthwise-separable convolution block. As in the encoder modules, time is folded into the batch dimension so the convolution cannot mix across timepoints:

$$[B, T, C', H, W] \;\rightarrow\; [B{\cdot}T, C', H, W] \;\xrightarrow{\text{conv}}\; [B{\cdot}T, C_\text{dec}, H, W] \;\rightarrow\; [B, T, C_\text{dec}, H, W]$$

This produces a spatial feature map $u \in \mathbb{R}^{B \times T \times C_\text{dec} \times H \times W}$ at each timepoint.

**Step 2 : FiLM modulation.**
The time-conditioned scale and shift are applied channel-wise to the spatial features. Broadcasting over $H$ and $W$:

$$y_{b,t,:,h,w} = \gamma_{b,t,:} \odot u_{b,t,:,h,w} + \beta_{b,t,:}$$

This is a lightweight but expressive operation : the spatial features are not changed structurally, but their magnitude and offset are rescaled in a time-dependent way. The network can learn, for example, to suppress certain feature channels at early timepoints and amplify them later, which is natural given the slow onset and decay of the hemodynamic response.

**Step 3 : Output projection and reshape.**
A final $1{\times}1$ convolution maps the modulated features to $7 \cdot L$ output channels, one set of 7 state variables per cortical layer:

$$[B{\cdot}T, C_\text{dec}, H, W] \;\rightarrow\; [B{\cdot}T, 7L, H, W]$$

The result is then reshaped and permuted into the final output layout:

$$[B, T, 7, L, H, W] \;\rightarrow\; [B, 7, L, T, H, W]$$



###### Optional: Time Derivatives
---

If `return_gradients=True`, the decoder additionally computes $\partial \hat{z} / \partial t$ at every timepoint. This is done analytically rather than with autograd over the full network. The key observation is that $u$ : the spatial features : does not depend on $t$ directly; only $\gamma$ and $\beta$ do. So by the chain rule:

$$\frac{\partial y}{\partial t} = \frac{\partial \gamma}{\partial t} \odot u + \frac{\partial \beta}{\partial t}$$

$\partial \gamma / \partial t$ and $\partial \beta / \partial t$ are computed using `jacrev` via `vmap` over individual scalar timepoints, which is exact and avoids differentiating through the entire encoder. This derivative is then pushed through the same $1{\times}1$ output convolution (which is linear, so it commutes with differentiation) to produce $\partial \hat{z} / \partial t \in \mathbb{R}^{B \times 7 \times L \times T \times H \times W}$.

These gradients are used in the PINN loss to enforce that the predicted state variables satisfy the hemodynamic differential equations at every timepoint.


---
The full forward pass through `HeinzleNet` can now be summarised as a clean sequence of these 4 modules:

- `MaskedLayerMixing` : encode laminar coupling
- `SpatialEncoder` : encode spatial structure independently at each timepoint
- `TemporalMixingEncoder` : encode temporal dynamics independently at each spatial location
- `SpatioTemporalDecoder` : decode to physical state variables, conditioned on continuous time

---

## FAQ: Why This Architecture?

This section addresses some natural questions about the design choices in `HeinzleNet`. If you are coming from a general deep learning background, some of these choices might look unusual at first.

---

**Why not a transformer?**

Transformers are a reasonable first thought for a problem with long temporal sequences, but they come with a significant practical problem: the self-attention mechanism scales quadratically with sequence length. With $T = 3000$ timepoints and $H \times W = 1024$ spatial locations, a naive spatiotemporal transformer would be completely intractable. Even restricting attention to the temporal axis alone , as in a standard sequence transformer , would require computing a $3000 \times 3000$ attention matrix for each spatial location in each batch, which is expensive at training time and even more so at the scale of full datasets.

The dilated TCN achieves a similar goal (relating distant timepoints) at linear cost in $T$, and the exponentially growing receptive field means you get global temporal context with far fewer parameters.

---

**Why not 3D convolutions?**

3D convolutions would process space and time jointly with a single $k \times k \times k$ kernel, which sounds appealing. The problem is that space and time are not the same kind of thing in this data. The spatial dimensions reflect the cortical layout of neural activity. The temporal dimension reflects a physiological process (the hemodynamic response) with a characteristic timescale of several seconds. Treating them symmetrically with a 3D kernel would mix very different signals and make it much harder for the network to learn structured representations of each.

The modular design , handle layers, then space, then time, separately, is a deliberate inductive bias. It also makes the model easier to interpret and debug, which matters when you want to connect its behaviour to known neuroscience.

---

**Why depthwise-separable convolutions instead of standard convolutions?**

The spatial grid is 32×32, which is small. A standard convolutional encoder with large channel counts would have far more parameters than the data can support, and would likely overfit or fail to generalise across subjects. Depthwise-separable convolutions factoise the spatial filtering and channel mixing into two cheaper operations, reducing parameter count by roughly a factor of $k^2$ (here $k = 3$, so about 9×) with negligible loss in expressivity. This is especially important in a PINN setting where the model is already constrained by physics-based losses that may reduce effective degrees of freedom.

---

**Why is the mask in `MaskedLayerMixing` fixed rather than learned?**

The mask encodes prior anatomical knowledge about which cortical layers can directly influence which others, based on the known draining hierarchy of the BOLD signal in laminar fMRI. Allowing the mask to be learned would let the network ignore this structure, which is something we specifically do not want , the whole point of MICH is to perform an inversion that respects the underlying physiology. A fixed mask is a hard constraint; a soft regularisation on the weights would be an alternative, but it is harder to interpret and easier for the network to circumvent during training.

---

**Why FiLM conditioning and not just concatenating $t$ as an extra channel?**

Concatenating $t$ as a scalar channel to the spatial feature map is simple but weak. It gives the network a single additional input value per pixel per timepoint, and any downstream use of that value has to be learned through potentially many convolutional layers. FiLM is more direct: it lets $t$ immediately rescale and shift the entire feature representation channel-wise, which is a much more expressive form of conditioning. It is also computationally cheap and has been shown to work well in physics-informed and conditional generation settings where the conditioning variable has a strong, global effect on the output.

---

**Why compute time derivatives analytically rather than using autograd through the full network?**

You could in principle call `torch.autograd.grad` on the full network output with respect to $t$ and get $\partial \hat{z} / \partial t$ that way. The problem is cost. This would require differentiating through all four modules , including the temporal encoder , which involves unrolling gradients through many convolutional layers and the entire sequence of TCN blocks. For a PINN training loop where these gradients are needed at every step, this is prohibitively expensive.

The analytic approach works because $t$ only enters the network through `FourierTimeEmbedding` → `TimeFiLM`. The spatial features $u$ are fixed given $x$, so the chain rule collapses to just differentiating $\gamma(t)$ and $\beta(t)$, which is a small MLP applied to a Fourier embedding. `jacrev` via `vmap` computes this efficiently per-timepoint without touching the rest of the network.

---

**What were the main alternatives considered for the overall architecture?**

A few natural alternatives come up in the literature for this kind of spatiotemporal inverse problem. A U-Net with 3D convolutions is a common baseline for volumetric biomedical data, but as noted above, the space-time symmetry assumption is wrong for fMRI. A purely recurrent architecture (LSTM or GRU) along the temporal axis is another option, but RNNs are slow to train on sequences of length 3000 due to their sequential nature, and they tend to struggle with very long-range dependencies compared to dilated convolutions. Purely spatial models that process each timepoint independently and ignore temporal structure entirely would fail to capture the slow dynamics of the hemodynamic response. The current design tries to take the best of each approach: structured spatial processing, efficient long-range temporal modelling, and physics-informed output conditioning.