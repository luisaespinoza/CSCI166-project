# 🎮 DQN Training Progression: Random Action Selection → Epsilon-Greedy → Custom Reward Logic

This project evaluates multiple Deep Q-Learning strategies on Atari **Tetris**, using both **Single DQN** and **Double DQN** architectures.  
Each training phase below was run and analyzed using both variants to compare stability, value estimation, and long-term performance.

---

## **1. Phase 1 — DQN With Pure Random Action Selection (500k Episodes)**  
In this baseline stage, we use a full DQN training setup (Single DQN and Double DQN), but override the policy so that **all actions are selected uniformly at random**, ignoring Q-values.

- DQN/DDQN still perform forward passes and updates  
- Replay buffer fills normally  
- Q-network learns from the resulting transitions  
- But the *behavior policy* is **pure random sampling**, not epsilon-greedy  

This phase shows how the networks behave under maximum exploration and provides a reference for later improvements.

---

## **2. Phase 2 — Standard Epsilon-Greedy DQN (2M Episodes)**  
Next, both **Single DQN** and **Double DQN** are trained using a classical **epsilon-greedy exploration schedule**.

- ε starts near 1.0 for strong exploration  
- ε decays gradually to favor exploitation  
- Replay buffer + target network stabilize learning  
- 2,000,000 training frames for each architecture  

This allows direct comparison of how DQN vs. DDQN diverge under the same exploration strategy.

---

## **3. Phase 3 — Epsilon-Greedy + Custom Reward/Penalty Logic (2M Episodes)**  
Finally, we introduce **custom reward shaping**, again testing both Single DQN and Double DQN:

- Penalties for unnecessary board height increases  
- Penalties for unstable/tower-like stack shapes  
- Rewards for line clears weighted by board stability or height reduction  
- Incentives for maintaining a smoother playable surface  

This stage encourages safer, longer-term strategies and reveals how each architecture responds to engineered reward signals.

---

# 🎥 Training Visualizations (GIF Previews)

Below are inline GIF previews stored in the repository, with placeholders for your Google Drive full-video links.

## Single DQN — 500k Episodes (Random Sampling)
![Single DQN 500k](tetris_gifs/singledqn_random.gif)  
[Full Video](https://drive.google.com/file/d/1CLyqZOrFKBArTFBeU1WJ0W5UIrJjS-tL/view?usp=sharing)

## Double DQN — 500k Episodes (Random Sampling)
![Double DQN 500k](tetris_gifs/dqn_random.gif)  
[Full Video](https://drive.google.com/file/d/1QJGR2Yj-QpTsmAxyrF_7eD3hYuOaA-fk/view?usp=sharing).

## Single DQN — 2M Episodes (Epsilon-Greedy)
![Single DQN 2M](tetris_gifs/singledqn_epsilon.gif)  
[Full Video](https://drive.google.com/file/d/1uxoF7FxUF459tFhoU96hrKLGH5I-sLjO/view?usp=sharing)

## Double DQN — 2M Episodes (Epsilon-Greedy)
![Double DQN 2M](tetris_gifs/double_dqn_epsilon.gif)  
[Full Video](https://drive.google.com/file/d/1WjvGx_EMZc3mgHx5z55HEJ9zJ_KL_ty9/view?usp=sharing)
