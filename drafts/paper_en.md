# \method{}: Learning Subject-to-Video Generation from Images Only

> Draft sections: Introduction, Related Work, Method. Citation keys (`\cite{}`) and the method name (`\method{}`) are placeholders. `Fig.~1` is the method overview (`fig/overview.tex`); `Sec.~X` refers to the experiments section to be written.

# Abstract

Subject-to-video generation renders a user-provided subject into a video described by text. Current systems are trained on paired reference–video data, which are scarce and biased: the reference is usually cropped from the target video, so training rewards pasting the reference into the frame rather than re-rendering the subject. We instead learn appearance from image pairs only, whose references deliberately differ from their targets, and borrow motion entirely from a frozen video diffusion transformer. Two mechanisms make static image supervision compatible with a video model. A signed motion adapter turns the backbone into a motion carrier whose scale $M$ moves continuously between a static regime, native motion, and amplified motion; running the carrier in its static regime during training lets static image targets shape appearance without teaching the model to be still. A paired-difference readout evaluates the reference branch with the real reference and with a learned null reference and adds only their difference to the carrier, so the reference never touches the motion computation and the reference-free branch for guidance is learned by construction. At inference, motion, text adherence, and reference adherence become three independent controls. On OpenS2V-Eval, with a copy-paste metric validated on synthetic anchors and gated by subject identity, the image-trained model copies the reference markedly less than the strongest video-trained baselines.

# 1. Introduction

Subject-to-video (S2V) generation renders a user-provided subject, given as a reference image, into a video described by a text prompt. Recent systems \cite{liu2025phantom, jiang2025vace, fei2025skyreelsa2, hu2025hunyuancustom, deng2025magref, xue2025standin} obtain this ability by fine-tuning a pretrained text-to-video backbone \cite{wan2025, kong2024hunyuanvideo} on paired reference–video data.

These pairs are the bottleneck. They are scarce, because every clip needs a clean, subject-centric reference whose identity is verified across the clip. More importantly, they are biased. The reference is usually cropped from a frame of the target video, so it shares pose, viewpoint, lighting, and often background with the target. Training on such pairs rewards a trivial solution: paste the reference into the frame. Recent benchmarks \cite{yuan2025opens2v, muss2025} document the resulting *copy-paste* behavior: the subject stays frozen in the reference pose, ignores prompted actions, and carries the reference lighting into the new scene.

Subject-driven image data suffer much less from either problem. Such datasets are orders of magnitude larger \cite{xiao2024omnigen, tan2024ominicontrol, wu2025uno}, and their reference–target pairs are constructed to differ in pose, scene, and viewpoint, so a model trained on them must re-render the subject rather than copy it. Our central proposition is therefore: **learn appearance from images, and let the video model contribute only motion.**

The obstacle is that images carry no motion. Supervising a video model with an image repeated into a static clip teaches it that the correct output is still, and the motion prior of the backbone is lost. Image-only training is useful only if static supervision can shape appearance without teaching the model to be static, and if appearance and motion remain separable afterwards.

\method{} meets both conditions on a frozen video diffusion transformer with two mechanisms (Fig.~1).

- **A motion carrier with a static regime.** A Stage-I low-rank adapter is trained with signed targets, static clips at $M{=}{+}1$ and real videos at $M{=}{-}1$, so that a runtime scale $M$ becomes a continuous motion control. During image training the carrier runs at $M{=}1$ and already produces the still video that the target demands. The appearance branch is therefore never asked to suppress motion, and setting $M{\le}0$ at inference restores it.
- **A paired-difference reference readout.** For every input, the appearance branch is evaluated once with the real reference and once with a learned null reference, and only the difference of the two predictions is added to the carrier. Everything that does not depend on the reference cancels, the carrier is never touched by reference-dependent computation, and the reference-free branch needed for guidance is learned by construction instead of by dropout.

As a result, motion, text adherence, and reference adherence become three independent controls at inference time.

Our contributions are as follows.

- We trace copy-paste to how paired video data are constructed, and show that subject-to-video generation can be trained from image pairs alone, with the pretrained video model serving purely as a source of motion.
- We introduce a signed motion carrier whose static regime absorbs static image supervision, so that image training shapes appearance without erasing motion.
- We introduce a paired real-versus-null difference readout that structurally separates the reference signal from the motion computation and yields independent motion, text, and reference controls.
- We propose a copy-paste metric that is simple to compute and validated on synthetic paste and natural re-render anchors, and report it together with a same-subject gate, so that low copying is not mistaken for failing to render the subject.

# 2. Related Work

**Subject-driven image generation.** Identity preservation in images has moved from per-subject optimization \cite{ruiz2023dreambooth, gal2022textual} and encoder-based adapters \cite{ye2023ipadapter, li2023blipdiffusion} to treating the reference as extra tokens of a diffusion transformer \cite{tan2024ominicontrol, xiao2024omnigen, wu2025uno}. The token-based formulation scales to large paired datasets whose references deliberately differ from their targets. We inherit this data regime and apply it to a video backbone.

**Subject-to-video generation.** Early identity-preserving video methods focus on faces \cite{he2024idanimator, yuan2024consisid}. Open-domain S2V methods condition a video backbone on arbitrary subjects by concatenating reference latents \cite{liu2025phantom, fei2025skyreelsa2}, through a unified conditioning interface \cite{jiang2025vace}, with multi-modal identity encoders \cite{hu2025hunyuancustom, deng2025magref}, or with a lightweight reference branch inside the backbone's self-attention \cite{xue2025standin}. All of them are trained on reference–video pairs, and OpenS2V-Nexus \cite{yuan2025opens2v} and MuSS \cite{muss2025} report the resulting copy-paste tendency. We reuse the reference branch of Stand-In \cite{xue2025standin}, but train it from images and add the two mechanisms that keep motion intact.

**Customizing video models from images.** AnimateDiff \cite{guo2023animatediff} attaches a separately trained motion module to customized image models. Closest to our Stage I, Still-Moving \cite{chefer2024stillmoving} trains a motion adapter on frozen videos to absorb the static bias of image training and removes it at inference. We differ in three ways: the adapter is trained with signed targets and becomes a continuous control rather than an on/off switch; it acts on a DiT with joint spatiotemporal attention; and instead of adapting the backbone, we keep it frozen and use the adapter as a fixed carrier for a separate appearance branch.

**Conditioning and guidance.** Classifier-free guidance \cite{ho2022cfg} and its multi-condition variants \cite{brooks2023instructpix2pix} contrast conditional and unconditional predictions; autoguidance \cite{karras2024autoguidance} contrasts a model with a weaker version of itself. In our readout the learned null reference is part of the training forward pass, not only an inference-time contrast.

# 3. Method

## 3.1 Overview

Given a reference image $r$ and a prompt $c$, we generate a video that shows the subject of $r$ and follows $c$. We use a pretrained text-to-video diffusion transformer, a modest set of captioned videos for Stage I only, and a large set of subject-driven image pairs $(r, x^{\mathrm{img}}, c)$ in which the target shows the subject of $r$ in a different pose, scene, or attribute. No reference–video pairs are used.

[[figure:overview]] Overview of \method{}. (a) Each denoising step runs three passes through the same frozen DiT; they share the noisy latent, timestep, and text, and share weights but not hidden states. The motion carrier adds the frozen Stage-I adapter scaled by $M$ and never reads the reference. The two reference passes read the real reference tokens $S$ or the learned null tokens $S_\varnothing$, and only the difference of their readouts is added to the carrier's prediction. During Stage II the carrier runs at $M{=}1$ and the target is a target image repeated as a still clip; at inference $M{\le}0$. (b) In a reference pass, video queries read video and reference keys in one softmax, and reference queries read only reference keys. (c) The Stage-I adapter is trained with signed targets, which turns $M$ into a continuous motion control.

The backbone is a latent video DiT \cite{wan2025, peebles2023dit}: a causal 3D VAE encodes the video, patches become tokens, and each of $L$ blocks applies timestep-modulated spatiotemporal self-attention with 3D RoPE, text cross-attention, and an FFN; $W_{\mathrm{out}}$ denotes the final projection to velocity. Training uses rectified flow \cite{liu2022rectified, lipman2022flow}: $z_\sigma = (1-\sigma) z_0 + \sigma\epsilon$ with target velocity $v^\star = \epsilon - z_0$ and logit-normal $\sigma$ \cite{esser2024sd3}.

\method{} keeps this backbone frozen and runs three passes through it at every denoising step (Fig.~1a). The prediction is
$$
\hat v = v_{\mathrm{mot}}(z_t, c; M) + \lambda\big[P(h_{\mathrm{real}}) - P(h_{\mathrm{null}})\big],
\tag{1}
$$
where $v_{\mathrm{mot}}$ is the prediction of the motion carrier, $h_{\mathrm{real}}$ and $h_{\mathrm{null}}$ are the final hidden states of the reference branch with the real and the learned null reference, and $P$ is a shared readout head. Sec.~3.2 builds the carrier, Sec.~3.3 and Sec.~3.4 the reference branch and its readout, and Sec.~3.5 explains why this combination lets a video model learn from images.

## 3.2 Stage I: A Signed Motion Carrier

**Adapter.** Following Still-Moving \cite{chefer2024stillmoving}, we attach rank-4 adapters to the query, key, and value projections of every self-attention layer and scale them by a runtime coefficient $M$:
$$
W_{\ast}\,u \;\mapsto\; W_{\ast}\,u + M\,\Delta W_{\ast}(u), \qquad \ast \in \{q, k, v\}.
\tag{2}
$$
The up-projection of each $\Delta W_\ast$ is zero-initialized, so $M{=}0$ recovers the backbone exactly.

**Signed training.** Still-Moving trains its adapter on frozen videos only and switches it off at inference. We instead train with two signs: static clips, made by repeating one image over all frames, are supervised at $M{=}{+}1$, and real videos at $M{=}{-}1$, both with the flow-matching loss over all frames. The adapter must then encode one direction in weight space that moves the backbone toward stillness when added and reproduces real motion when subtracted. $M$ thus becomes a continuous control (Fig.~1c): $M{=}1$ gives still output, $M{=}0$ the backbone's native motion, and $M{<}0$ amplified motion. This control has two uses. At inference it sets the amount of motion; during Stage II its static end $M{=}1$ is exactly the regime that static image targets require (Sec.~3.5). After Stage I the adapter is frozen; together with the backbone it forms the *motion carrier*.

## 3.3 Reference Branch with a Learned Null

The reference branch follows Stand-In \cite{xue2025standin} and is not our contribution; we summarize it. It reuses the frozen backbone with the motion adapter disabled ($M{=}0$). The reference image is encoded by the same VAE and patch embedding into tokens $S$ with timestep $0$, placed at 3D RoPE coordinates disjoint from the video (temporal index $-1$, spatial offset outside the video grid). Reference tokens attend only to themselves through a rank-128 LoRA on their query, key, and value projections, and video queries read them in one joint softmax (Fig.~1b):
$$
O_v = \mathrm{softmax}\!\Big(\tfrac{1}{\sqrt d}\,Q_v [K_v; K_r]^{\top} + \mathbf{m}\Big)[V_v; V_r],
\tag{3}
$$
where $\mathbf{m}$ masks reference tokens that cover padding.

**Learned null reference.** Our addition to the branch is a learnable token $e_\varnothing$, broadcast over the reference token grid to form a null reference $S_\varnothing$. It follows exactly the same path as a real reference: the same positions, the same reference self-attention, and the same joint attention. A real and a null pass therefore differ only in the content of the reference tokens, never in the computation, which is what makes the readout below well defined.

## 3.4 Paired-Difference Readout

For a given noisy latent, timestep, and text, the reference branch runs twice with shared parameters and separate hidden states, once with $S$ and once with $S_\varnothing$. After the backbone's final normalization and modulation, both final hidden states go through the same head, and only their difference is used:
$$
\begin{aligned}
P(h) &= \beta\,W_{\mathrm{out}}(h) + W_{\Delta}(h),\\
\Delta v &= P(h_{\mathrm{real}}) - P(h_{\mathrm{null}}),
\end{aligned}
\tag{4}
$$
where $W_{\mathrm{out}}$ is the frozen output projection, $\beta{=}0.1$, and $W_\Delta$ is a trainable linear layer initialized to zero. The fused prediction is Eq.~(1) with $\lambda{=}1$.

This construction is the core of the method, and four properties follow from it.

- **Only reference-dependent signal.** Any component that does not depend on the reference, such as content determined by the text alone or the bias of the head, is identical in both passes and cancels. A single-pass correction $P(h_{\mathrm{real}})$ would carry such content as well and compete with the carrier for it.
- **Structural decoupling.** The carrier never consumes reference tokens or reference-dependent hidden states. For a fixed input, $v_{\mathrm{mot}}$ is independent of the reference and of every Stage-II parameter, so $M$ acts only on motion and the reference acts only through $\Delta v$.
- **A free unconditional branch.** The null pass is part of every training example, so no reference dropout is needed; the reference-free branch used for guidance is trained by construction.
- **Gradient from the first step.** The frozen term $\beta W_{\mathrm{out}}$ gives a non-zero readout while $W_\Delta$ is still zero, so the reference LoRA receives gradient from the first update.

## 3.5 Learning from Images

**Static targets.** Each training example is an image pair $(r, x^{\mathrm{img}}, c)$. The target image is repeated along time into a static clip and encoded into $z_0$, which gives a video-shaped target without any video data.

**Absorbing the static bias.** The key step is to run the motion carrier at $M{=}1$, its static regime, during Stage II, with gradients disabled. The carrier then already predicts a still video, which is what the static target demands. The residual that the reference branch has to supply is the appearance difference between a generic still video and the target, not the suppression of motion. Static supervision therefore shapes appearance without teaching the model to be static, and evaluating the carrier at $M{\le}0$ at inference restores the backbone's motion.

**Objective.** Only the fused prediction is supervised:
$$
\mathcal{L} = \mathbb{E}\,\big\lVert \hat v - v^{\star} \big\rVert_2^2 + \eta\,\mathcal{L}_{\mathrm{repa}},
\tag{5}
$$
averaged over all latent frames. Through Eq.~(4), the real pass receives the gradient with a positive sign and the null pass with a negative sign, so the two jointly learn the increment that a real reference should contribute, and $e_\varnothing$ is trained as the reference-free baseline. Two standard regularizers complete the objective: text dropout in all three passes for text guidance, and a small REPA term \cite{yu2024repa} that aligns reference tokens at intermediate blocks of the real pass with frozen DINOv3 patch features \cite{simeoni2025dinov3}. The trainable parameters are the reference LoRA, $e_\varnothing$, $W_\Delta$, and the REPA projectors.

## 3.6 Inference

Text and reference adherence get separate scales:
$$
\begin{aligned}
\hat v ={}& \hat v(c_\varnothing, S_\varnothing)\\
&+ \omega_T\big[\hat v(c, S_\varnothing) - \hat v(c_\varnothing, S_\varnothing)\big]\\
&+ \omega_I\big[\hat v(c, S) - \hat v(c, S_\varnothing)\big],
\end{aligned}
\tag{6}
$$
where $c_\varnothing$ is the negative prompt. In the two null-reference terms $\Delta v$ is zero by Eq.~(4), so they reduce to the carrier's prediction. The $\omega_I$ term therefore scales exactly the reference correction and the $\omega_T$ term acts on text alone, while $M$ is set independently and may vary per block or over denoising steps. Because the carrier does not depend on the reference, Eq.~(6) needs four backbone passes per step instead of nine: two carrier passes and one real and one null reference pass. Autoguidance \cite{karras2024autoguidance} with an earlier checkpoint of the reference branch can be added as an optional quality term.

## 3.7 Discussion

The decoupling of Sec.~3.4 holds for one forward evaluation: with the same latent, timestep, text, and $M$, the carrier does not depend on the reference. It does not hold for a whole sampling trajectory, because the fused prediction updates the latent that later carrier evaluations read. The architecture prevents the reference from rewriting the motion computation; it does not, and should not, prevent the subject's appearance from influencing how the scene evolves.

The correction $\Delta v$ is fitted while the carrier is static and is applied while it moves. The construction therefore suits appearance that is shared across frames, which is what image pairs can supervise, and leaves temporal variation to the carrier. How well $\Delta v$ transfers across $M$ is evaluated in Sec.~X.
