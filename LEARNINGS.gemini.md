> **Historical notes.** Written early in the project (Gemini-generated); several details are superseded (e.g. the 14-d SAC state, the Cornell dataset, a pick-and-place SAC task, the <1 ms ISR claim). Current design: `architecture.md`; measured results: `report.md`, `docs/FULL_PROJECT_REPORT.md`, `docs/JETSON_RESULTS.md`.

# Learning Log & Technical Notes (`LEARNINGS.gemini.md`)

## 1. Reinforcement Learning for 4-DOF Grasping Pipeline
- **Algorithm**: Soft Actor-Critic (SAC). Chosen for off-policy sample efficiency and maximum entropy formulation, encouraging exploration in continuous joint torque space.
- **State Space**: $S = [\theta_1, \theta_2, \theta_3, \theta_4, \dot{\theta}_1, \dot{\theta}_2, \dot{\theta}_3, \dot{\theta}_4, x_{ee}, y_{ee}, z_{ee}, x_{tgt}, y_{tgt}, z_{tgt}]$.
- **Action Space**: Continuous joint commands $A \in [-1.0, 1.0]^4$ mapped to motor torques/velocities.
- **Reward Function**: 
  $$R_t = - w_{\text{dist}} \| \mathbf{p}_{ee} - \mathbf{p}_{tgt} \|_2 - w_{\text{action}} \| \mathbf{a}_t \|_2 - w_{\text{jerk}} \| \mathbf{a}_t - \mathbf{a}_{t-1} \|_2 + R_{\text{reach}}$$
  where $R_{\text{reach}} = +100$ when $\| \mathbf{p}_{ee} - \mathbf{p}_{tgt} \|_2 < 0.03 \text{ m}$.

## 2. Few-Shot Perception & Out-of-Distribution (OOD) Uncertainty
- **Backbone**: MobileNetV3 (Small/Large) for low-latency feature extraction on edge platforms like Jetson Orin Nano.
- **Mahalanobis Distance Metric**:
  Given feature vector $\mathbf{z} = f_{\theta}(\mathbf{x}) \in \mathbb{R}^d$, class conditional mean $\boldsymbol{\mu}_c$, and tied covariance matrix $\boldsymbol{\Sigma}$:
  $$D_M(\mathbf{x}) = \min_c \sqrt{(\mathbf{z} - \boldsymbol{\mu}_c}^T \boldsymbol{\Sigma}^{-1} (\mathbf{z} - \boldsymbol{\mu}_c)}$$
  If $D_M(\mathbf{x}) > \tau_{\text{uncertainty}}$, the target object is classified as out-of-distribution / high uncertainty.

## 3. Edge Deployment & Hardware Safety
- **TensorRT Compilation**: Exports PyTorch $\rightarrow$ ONNX $\rightarrow$ TensorRT FP16 plan file for hardware acceleration on Jetson Orin Nano.
- **ROS2 & micro-ROS Inter-Process Bridge**:
  - `evidential_vision_node` publishes `/arm_target` (`[x, y, z, class_id, uncertainty]`).
  - ESP32 subscribes via micro-ROS over Serial/UDP.
  - If `uncertainty > 0.75`, ESP32 immediately fires a hardware interrupt (ISR) to disable motor drivers and halt movement to prevent collisions.

## 4. Gymnasium Step Truncation & SAC Anti-Overfitting Callback
- **Environment Truncation Bug**: When implementing `Gymnasium.Env`, `self.current_step` MUST be reset to `0` inside `reset()`. Failing to reset step counter causes subsequent episode rollouts to immediately satisfy `current_step >= max_steps`, leading to 1-step episode truncations (`ep_len_mean = 1`).
- **EvalCallback & Dynamic Entropy Regularization**: Wrapping training with `EvalCallback` evaluating deterministic performance over randomized test seeds prevents over-fitting to training trajectories. Saving checkpoints based on peak evaluation returns guarantees saving the optimal policy weight matrix (`best_model.zip`).

