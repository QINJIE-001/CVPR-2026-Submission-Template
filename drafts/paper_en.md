# \method{}: Learning Subject-to-Video Generation from Images Only

> Draft sections: Introduction, Related Work, Method. Citation keys (`\cite{}`) and the method name (`\method{}`) are placeholders. `Fig.~X` / `Sec.~X` refer to figures and sections to be added.

# 1. Introduction

Subject-to-video (S2V) generation aims to synthesize a video that depicts a user-specified subject, given a reference image of that subject and a text prompt. The task sits at the intersection of two capabilities that modern diffusion models have acquired separately: faithful subject preservation, which has matured in the image domain \cite{ruiz2023dreambooth, ye2023ipadapter, tan2024ominicontrol, wu2025uno}, and temporally coherent motion synthesis, which is now provided by large pretrained text-to-video models \cite{wan2025, kong2024hunyuanvideo}. Recent S2V systems \cite{liu2025phantom, jiang2025vace, fei2025skyreelsa2, hu2025hunyuancustom, deng2025magref, xue2025standin} bring the two together by fine-tuning a video backbone on paired reference–video data.

This recipe is limited by its data. First, paired reference–video data are scarce and expensive to collect: every training clip must be annotated with a clean, subject-centric reference whose identity is verified across the clip. Second, and more fundamentally, the pairs that *can* be collected at scale are biased in a way that harms the learned behavior. In practice the reference is cropped from a frame of the same video, so it shares pose, viewpoint, lighting, and even background with the target. A model trained on such pairs is rewarded for a trivial solution: paste the reference into the frame. This *copy-paste* behavior has been documented by recent benchmarks \cite{yuan2025opens2v, muss2025} and manifests as subjects that are frozen in the reference pose, fail to follow prompted actions, and transfer reference lighting and background into the generated scene.

Image data do not suffer from either problem to the same degree. Subject-driven *image* datasets are orders of magnitude larger \cite{xiao2024omnigen, tan2024ominicontrol, wu2025uno}, and their reference–target pairs are constructed to vary: the same subject appears in a different pose, scene, viewpoint, or with altered attributes. A model trained on them must preserve identity while *re-rendering* the subject, rather than copying it. This leads to the central proposition of this paper: **appearance should be learned from images, and the video model should contribute only motion.**

Realizing this proposition is not straightforward, because images carry no temporal information. If a video model is naively supervised with images (for example, by repeating an image into a static clip), the model learns that the correct output is a still video, and the motion prior of the backbone is destroyed in the process. Any image-only training scheme for S2V therefore has to satisfy three requirements simultaneously: (i) motion must be *borrowed* from the pretrained video model, since it cannot be learned from images; (ii) appearance must be learned from image pairs; and (iii) the two must be *decoupled*, so that static image supervision shapes appearance without overwriting motion, and motion can be recovered freely at inference time.

We propose \method{}, a two-stage framework built on a frozen video diffusion transformer that meets these requirements by construction. Stage I turns the frozen backbone into a *controllable motion carrier*: a low-rank adapter on the spatiotemporal self-attention, in the spirit of Still-Moving \cite{chefer2024stillmoving}, is trained so that a runtime scale $M$ interpolates between a static regime ($M{=}1$), the native motion of the backbone ($M{=}0$), and amplified motion ($M{<}0$). Stage II learns appearance from image pairs through a reference branch that injects reference tokens into the backbone's self-attention, following Stand-In \cite{xue2025standin}. The key to decoupling is twofold. During Stage II training the motion carrier is run at $M{=}1$, so the carrier itself already produces the static video that the image supervision demands; the reference branch is therefore never asked to suppress motion. In addition, the reference contribution is read out as a *paired difference*: for every input, the reference branch is evaluated once with the real reference and once with a learned null reference, and only the difference between the two predictions is added to the carrier's output. Everything that does not depend on the reference cancels, and the motion carrier is never touched by reference-dependent computation within a forward pass. At inference, $M$ is set to zero or negative to restore motion, and text and reference guidance are applied as independent terms.

Our contributions are as follows.

- We identify copy-paste as a consequence of how paired video data are constructed, and propose to train subject-to-video generation from image pairs alone, using the pretrained video model purely as a source of motion.
- We introduce a decoupled architecture in which a controllable motion carrier and an image-trained reference branch share a frozen backbone but maintain independent hidden states, and in which the reference signal is extracted through a paired real-versus-null difference readout.
- We show that running the motion carrier in its static regime during training lets a video model absorb static image supervision without losing its motion prior, and that motion, text adherence, and reference adherence become three independently adjustable controls at inference time.

# 2. Related Work

**Subject-driven image generation.** Preserving the identity of a user-provided subject in generated images has been addressed by per-subject optimization \cite{ruiz2023dreambooth, gal2022textual}, by encoder-based adapters that inject image features through cross-attention \cite{ye2023ipadapter, li2023blipdiffusion}, and more recently by treating the reference as additional tokens of a diffusion transformer \cite{tan2024ominicontrol, xiao2024omnigen, wu2025uno}. The token-based formulation scales to large paired datasets whose references deliberately differ from their targets in pose, scene, and attributes \cite{tan2024ominicontrol, xiao2024omnigen}. Our work inherits this data regime and this injection mechanism, but applies them to a *video* backbone.

**Subject-to-video generation.** Early identity-preserving video methods focused on faces, using identity embeddings as conditions \cite{he2024idanimator, yuan2024consisid}. Open-domain S2V systems condition a video backbone on arbitrary subjects, either by concatenating reference latents along the sequence \cite{liu2025phantom, fei2025skyreelsa2}, by a unified conditioning interface \cite{jiang2025vace}, by multi-modal identity encoders \cite{hu2025hunyuancustom, deng2025magref}, or by a lightweight reference branch that shares the backbone's self-attention \cite{xue2025standin}. All of these are trained on reference–video pairs. OpenS2V-Nexus \cite{yuan2025opens2v} highlights the resulting copy-paste tendency and evaluates naturalness alongside identity, and MuSS \cite{muss2025} proposes a copy-paste rate to quantify it. We adopt Stand-In's reference branch \cite{xue2025standin} as the appearance pathway, but replace video supervision with image supervision and add the mechanisms required to keep motion intact under that supervision.

**Customizing video models from images.** AnimateDiff \cite{guo2023animatediff} showed that a motion module trained once can be attached to image models customized on image data. Still-Moving \cite{chefer2024stillmoving} addressed the converse problem: adapting a video model's spatial layers on images while keeping its temporal layers, using a motion adapter trained on frozen videos to absorb the static bias of image training and then removed at inference. Our Stage I generalizes this idea to a DiT whose attention is jointly spatiotemporal, and extends it with a signed training regime so that the adapter scale becomes a continuous motion control rather than an on/off switch. Unlike Still-Moving, which adapts the backbone itself, we keep the backbone frozen and use the motion adapter as a fixed carrier on top of which a separate appearance branch is trained.

**Decoupled conditioning and guidance.** Classifier-free guidance \cite{ho2022cfg} contrasts conditional and unconditional predictions; multi-condition variants apply separate guidance weights to separate conditions \cite{brooks2023instructpix2pix}. Autoguidance \cite{karras2024autoguidance} contrasts a model with a weaker version of itself to improve quality without strengthening the condition. Our paired-difference readout uses a learned null reference not only for guidance at inference but as a *training-time* construction that isolates the reference-dependent component of the prediction. Representation alignment \cite{yu2024repa} regularizes diffusion features toward a frozen visual encoder; we apply it to the reference tokens only.

# 3. Method

## 3.1 Problem Setting and Overview

Given a reference image $r$ and a text prompt $c$, we aim to generate a video $x \in \mathbb{R}^{F\times H\times W\times 3}$ that depicts the subject in $r$ and follows $c$. We assume access to (a) a pretrained text-to-video diffusion transformer, (b) a modest set of videos with captions, used only to learn a motion control, and (c) a large set of subject-driven *image* pairs $(r, x^{\mathrm{img}}, c)$, in which the target image shows the same subject as $r$ under a different pose, scene, or attribute. No reference–video pairs are used.

\method{} keeps the backbone frozen and adds two adapters that are trained in two stages (Fig.~X). Stage I produces a *motion carrier* whose motion strength is controlled by a scalar $M$. Stage II trains a *reference branch* on image pairs. At every denoising step the final velocity prediction is
$$
\hat v = v_{\mathrm{motion}}(z_t, c; M) \;+\; \lambda\,\big[P(h_{\mathrm{real}}) - P(h_{\mathrm{null}})\big],
\tag{1}
$$
where $v_{\mathrm{motion}}$ is the carrier's prediction, $h_{\mathrm{real}}$ and $h_{\mathrm{null}}$ are the final hidden states of the reference branch evaluated with the real reference and with a learned null reference, respectively, and $P$ is a shared readout head. The three terms are computed by three passes through the same frozen backbone that share parameters but not hidden states.

## 3.2 Preliminaries

**Backbone.** We build on a latent video DiT \cite{wan2025, peebles2023dit}. A video is encoded by a causal 3D VAE into latents $z \in \mathbb{R}^{f\times h\times w\times C}$, patchified into $N_v$ tokens, and processed by $L$ transformer blocks. Each block applies timestep-modulated self-attention over all spatiotemporal tokens, cross-attention to text embeddings, and a feed-forward network:
$$
\begin{aligned}
u &= \mathrm{LN}_1(X)\,(1+s_1(t)) + b_1(t), \\
X_1 &= X + g_1(t)\, W_o\,\mathrm{Attn}(W_q u, W_k u, W_v u), \\
X_2 &= X_1 + \mathrm{CrossAttn}(\mathrm{LN}_2(X_1), c), \\
X_3 &= X_2 + g_3(t)\,\mathrm{FFN}\big(\mathrm{LN}_3(X_2)(1+s_3(t)) + b_3(t)\big),
\end{aligned}
\tag{2}
$$
with 3D rotary position embeddings applied to queries and keys. We denote by $W_{\mathrm{out}}$ the final projection that maps the normalized last hidden state to the velocity.

**Flow matching.** Training uses the rectified-flow objective \cite{liu2022rectified, lipman2022flow}. For clean latents $z_0$, noise $\epsilon \sim \mathcal{N}(0, I)$, and noise level $\sigma \in (0,1)$ drawn from a logit-normal distribution \cite{esser2024sd3}, the noisy input is $z_\sigma = (1-\sigma) z_0 + \sigma \epsilon$ and the target velocity is $v^\star = \epsilon - z_0$.

## 3.3 Stage I: A Controllable Motion Carrier

The role of Stage I is to make the backbone's motion *adjustable*, and in particular to give it a static regime that can later absorb image supervision.

**Motion adapter.** Following Still-Moving \cite{chefer2024stillmoving}, we attach low-rank adapters \cite{hu2022lora} to the self-attention of every block. Because the backbone's self-attention is jointly spatiotemporal, the adapters are placed on the query, key, and value projections of this attention, scaled by a runtime coefficient $M$:
$$
Q = W_q u + M\,\Delta W_q(u),\qquad
K = W_k u + M\,\Delta W_k(u),\qquad
V = W_v u + M\,\Delta W_v(u).
\tag{3}
$$
Each $\Delta W$ is a rank-$r$ factorization ($r{=}4$) with a zero-initialized up-projection, so that $M{=}0$ recovers the backbone exactly at initialization and at any later time.

**Signed training.** Still-Moving trains the adapter on frozen videos only, so that it learns to *remove* motion and is switched off at inference. We instead train with two kinds of clips and two signs of $M$: static clips obtained by repeating a single image across all frames are supervised with $M{=}{+}1$, and real videos are supervised with $M{=}{-}1$. Under the flow-matching loss over all frames, the adapter must therefore encode a direction in parameter space that moves the backbone from its native motion toward stillness when added, and that reproduces real motion when subtracted. The result is a continuous control: $M{=}1$ yields static output, $M{=}0$ yields the backbone's native motion, and $M{<}0$ amplifies motion. Only the adapter is trained; the backbone, VAE, and text encoder are frozen.

After Stage I the adapter is frozen. In everything that follows, "the motion carrier" refers to the frozen backbone together with this frozen adapter, evaluated at a chosen $M$.

## 3.4 Stage II: Learning Appearance from Image Pairs

The reference branch adds the subject's appearance. It reuses the frozen backbone with the motion adapter *disabled* ($M{=}0$), so that this branch models appearance only.

**Reference tokens.** The reference image is encoded by the same VAE and patch embedding as the video, producing $N_r$ reference tokens $S$. Following Stand-In \cite{xue2025standin}, the reference is treated as a clean conditioning signal: its timestep is fixed to $0$, and the modulation parameters in Eq.~(2) are computed from $t{=}0$ for reference tokens and from the actual $t$ for video tokens. Reference tokens pass through all $L$ blocks alongside the video tokens and are discarded before the output head.

**Conditional position mapping.** Reference tokens receive rotary coordinates disjoint from the video: the temporal coordinate of the $i$-th reference image is $-(i{+}1)$, and its spatial coordinates are offset beyond the video grid, i.e., $h \in [h_v, h_v + h_r)$ and $w \in [w_v, w_v + w_r)$. The backbone can thus distinguish reference from video tokens using only positional information, and several reference images can be supplied without ambiguity.

**Reference self-attention.** Reference tokens attend only to themselves. Their query, key, and value projections receive a separate low-rank adapter of rank $128$:
$$
Q_r = (W_q + \Delta W^{\mathrm{ref}}_q)\,u_r,\quad
K_r = (W_k + \Delta W^{\mathrm{ref}}_k)\,u_r,\quad
V_r = (W_v + \Delta W^{\mathrm{ref}}_v)\,u_r,
\tag{4}
$$
after which the reference tokens pass through the frozen cross-attention and FFN. Because reference tokens never read video tokens, the reference stream is a stable conditioning source whose content does not drift with the noisy video.

**Joint attention from video to reference.** Video queries attend to video keys and reference keys in a single softmax:
$$
A = \mathrm{softmax}\!\left(\frac{[\,Q_v K_v^{\top}\;\big|\;Q_v K_r^{\top}\,]}{\sqrt d} + \mathbf{m}\right),\qquad
O = A_v V_v + A_r V_r,
\tag{5}
$$
where $A_v$ and $A_r$ are the column blocks of $A$ and $\mathbf{m}$ masks reference tokens that correspond to padding introduced when the reference is letterboxed to the video resolution. The reference is thus exposed to the backbone as additional tokens that its existing attention can read; no extra fusion module or gate is introduced. The reference keys and values in Eq.~(5) are the same tensors used in Eq.~(4).

**Learned null reference.** We introduce a learnable token $e_\varnothing \in \mathbb{R}^{C}$ that is broadcast over the reference token grid to form a *null reference* $S_\varnothing$. It is processed by exactly the same path as a real reference, including position mapping, reference self-attention, and joint attention. The null reference therefore differs from a real one only in content, not in computation, which is what makes the difference readout of Sec.~3.5 meaningful.

## 3.5 Paired-Difference Readout

For a given noisy latent $z_\sigma$, timestep, and text, the reference branch is evaluated twice with shared parameters and independent hidden states: once with the real reference $S$, yielding final hidden state $h_{\mathrm{real}}$, and once with the null reference $S_\varnothing$, yielding $h_{\mathrm{null}}$. Each hidden state is passed through the backbone's final normalization and timestep modulation, then through a readout head
$$
P(h) = \beta\, W_{\mathrm{out}}(h) + W_\Delta(h),
\tag{6}
$$
where $W_{\mathrm{out}}$ is the frozen output projection of the backbone, $\beta$ is a small constant ($0.1$), and $W_\Delta$ is a trainable linear layer initialized to zero. The reference correction is the difference
$$
\Delta v = P(h_{\mathrm{real}}) - P(h_{\mathrm{null}}),
\tag{7}
$$
and the final prediction is Eq.~(1) with $\lambda{=}1$.

Three properties follow from this construction. First, any component of the prediction that does not depend on the reference, including content determined by the text alone and the bias of the readout head, is identical in both passes and cancels in Eq.~(7); $\Delta v$ carries only reference-dependent information. Second, the frozen term $\beta W_{\mathrm{out}}$ provides a non-zero readout and a gradient path at the start of training, when $W_\Delta$ is still zero, so that the reference adapters receive learning signal from the first update; $W_\Delta$ then learns a correction to this fixed readout. Third, the motion carrier never consumes reference tokens or reference-dependent hidden states. For a fixed input, $v_{\mathrm{motion}}$ is independent of the reference and of all Stage II parameters, so the motion scale $M$ acts only on motion and the reference acts only through $\Delta v$.

## 3.6 Training the Reference Branch from Images

**Static supervision.** Each training example is an image pair $(r, x^{\mathrm{img}}, c)$. The target image is repeated along the temporal axis to form a static clip, which is encoded by the video VAE into $z_0$. This provides a video-shaped target without any video data.

**Absorbing the static bias.** The motion carrier is evaluated at $M{=}1$ during training, in its static regime, with gradients disabled. In this regime the carrier already predicts a still video, which is exactly what the static target demands. The residual that the reference branch must supply is therefore the appearance difference between a generic still video and the target, not the suppression of motion. This is what allows static image supervision to shape appearance without teaching the model to be static. At inference the carrier is evaluated at $M \le 0$, and the backbone's motion prior is restored.

**Objective.** Only the fused prediction is supervised:
$$
\mathcal{L}_{\mathrm{diff}} = \mathbb{E}\big[\,\|\hat v - v^\star\|_2^2\,\big],
\tag{8}
$$
averaged over all latent frames. Neither $P(h_{\mathrm{real}})$ nor $P(h_{\mathrm{null}})$ is required to match the target on its own. Through Eq.~(7), the real pass receives the gradient of $\mathcal{L}_{\mathrm{diff}}$ with positive sign and the null pass with negative sign, so the two passes jointly learn the increment that a real reference should contribute, and the null token is trained as the reference-free baseline. Because the null pass is part of every training example, no stochastic reference dropout is required; the unconditional reference branch needed for guidance is learned structurally.

**Text dropout.** With probability $p$, the text embedding is replaced by the embedding of the empty prompt in all three passes, providing the text-unconditional prediction used by classifier-free guidance \cite{ho2022cfg}.

**Representation alignment.** To regularize the reference stream, the reference tokens at a set of intermediate blocks $\mathcal{B}$ of the real pass are projected by per-block MLPs $\phi_\ell$ and aligned to the patch features of a frozen self-supervised encoder $E$ \cite{simeoni2025dinov3} applied to the reference image, as in REPA \cite{yu2024repa}:
$$
\mathcal{L}_{\mathrm{repa}} = -\frac{1}{|\mathcal{B}|}\sum_{\ell \in \mathcal{B}} \mathbb{E}_{\mathrm{patch}}\big[\cos\big(\phi_\ell(S^{(\ell)}),\, E(r)\big)\big],
\tag{9}
$$
with padding patches excluded. The total loss is $\mathcal{L} = \mathcal{L}_{\mathrm{diff}} + \eta\,\mathcal{L}_{\mathrm{repa}}$ with a small $\eta$. The trainable parameters of Stage II are the reference adapters $\Delta W^{\mathrm{ref}}$, the null token $e_\varnothing$, the readout $W_\Delta$, and the projectors $\phi_\ell$; the backbone, motion adapter, VAE, text encoder, and $E$ remain frozen.

## 3.7 Inference

**Independent guidance.** Text and reference adherence are controlled by separate scales $\omega_T$ and $\omega_I$ through a three-branch guidance rule:
$$
\hat v = \hat v(c_\varnothing, S_\varnothing)
+ \omega_T\big[\hat v(c, S_\varnothing) - \hat v(c_\varnothing, S_\varnothing)\big]
+ \omega_I\big[\hat v(c, S) - \hat v(c, S_\varnothing)\big],
\tag{10}
$$
where $c_\varnothing$ denotes the negative (or empty) prompt. For the two null-reference branches, $\Delta v$ is identically zero by Eq.~(7), so they reduce to the motion carrier's prediction; the $\omega_I$ term therefore amplifies precisely the reference correction, and the $\omega_T$ term acts on text alone. The motion scale $M$ enters all branches through the carrier and is set independently; it may be constant, specified per block, or scheduled over denoising steps.

**Optional autoguidance.** A fourth branch $\omega_W[\hat v(c,S) - \hat v_{\mathrm{weak}}(c,S)]$ may be added, where $\hat v_{\mathrm{weak}}$ uses the reference adapters of an earlier checkpoint while sharing the backbone and motion carrier \cite{karras2024autoguidance}. Both branches carry identical conditions, so this term sharpens quality without strengthening either condition.

**Efficient evaluation.** The three branches of Eq.~(10) nominally require nine backbone passes. Since the motion carrier does not depend on the reference, its two positive-prompt evaluations coincide, and since the null-reference branches contribute no correction, only one real and one null reference pass are needed. Each denoising step thus costs four passes: two carrier passes (negative and positive prompt) and one real and one null reference pass.

## 3.8 Discussion

The decoupling established in Sec.~3.5 is a statement about a single forward evaluation: given the same noisy latent, timestep, text, and $M$, the carrier's hidden states and prediction are independent of the reference. It is not a guarantee about an entire sampling trajectory. The fused prediction updates the latent at each step, and the reference therefore influences the input of subsequent carrier evaluations. The architecture guarantees that reference parameters cannot *rewrite* the motion computation; it does not, and should not, prevent the subject's appearance from affecting how the scene evolves.

A second remark concerns what the reference branch learns under static supervision. Because the carrier is static during training, the correction $\Delta v$ is fitted on targets without temporal variation and is applied, at inference, to a carrier that moves. The construction is therefore well suited to appearance that is shared across frames, which is the component that image pairs can supervise, and leaves temporal variation entirely to the carrier. Whether $\Delta v$ transfers cleanly across values of $M$ is an empirical question that we address in Sec.~X.
