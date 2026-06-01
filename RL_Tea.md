# Perfect Problem-Solving Study Guide — Reinforcement Learning
> **Coverage:** Lectures 1–10 | **Supplemented by:** Midterm Practice Revision PDF  
> All numeric examples are extracted verbatim from lecture slides and the revision PDF, with full step-by-step workings.

---

## Table of Contents
1. [TD State-Value Update](#1-td-state-value-update)
2. [Greedy Action Selection](#2-greedy-action-selection-mid) (MID)
3. [ε-Greedy Action Selection](#3-ε-greedy-action-selection-mid) (MID)
4. [Stationary Action-Value Estimation](#4-stationary-action-value-estimation-mid) (MID)
5. [Non-Stationary Action-Value Estimation](#5-non-stationary-action-value-estimation)
6. [Softmax Action Selection](#6-softmax-action-selection-mid) (MID)
7. [Upper-Confidence-Bound (UCB) Selection](#7-upper-confidence-bound-ucb-selection)
8. [Gradient Bandit / Action Preferences](#8-gradient-bandit-action-preferences-mid) (MID)
9. [MRP / Bellman Equation — State Value & Transition Probability](#9-mrp-bellman-equation-state-value-transition-probability-mid) (MID)
10. [Policy Evaluation (Iterative DP) — Gridworld](#10-policy-evaluation-iterative-dp-gridworld)
11. [Policy Improvement — Q-value Computation](#11-policy-improvement-q-value-computation)
12. [Asynchronous DP — In-Place Updates](#12-asynchronous-dp-in-place-updates)
13. [Monte Carlo Prediction (State-Value Update)](#13-monte-carlo-prediction-state-value-update)
14. [Monte Carlo Action-Value (Q-table) Update](#14-monte-carlo-action-value-q-table-update)
15. [Monte Carlo Control with ε-Soft Policy](#15-monte-carlo-control-with-ε-soft-policy)
16. [TD(0) State-Value Update](#16-td0-state-value-update)
17. [SARSA — On-Policy TD Control](#17-sarsa-on-policy-td-control)
18. [Q-Learning — Off-Policy TD Control](#18-q-learning-off-policy-td-control)
19. [n-Step TD Return Calculation](#19-n-step-td-return-calculation)
20. [Complex Return TD(λ)](#20-complex-return-tdλ)
21. [DQN — TD Target & Loss Calculation](#21-dqn-td-target-loss-calculation)

---

## 1. TD State-Value Update

**Problem Name:** Temporal-Difference State-Value Update (Gradient-Based)

**Problem Definition:**  
Given a sequence of states $s_1 \to s_2 \to \cdots \to s_T$ with known outcomes, update each state's estimated win-probability using TD learning. No environment model is needed.

**Formula/Approach:**  
$$V(s) \leftarrow V(s) + \alpha \big[V(s') - V(s)\big]$$
- $\alpha$ = learning rate  
- $V(s')$ = value of the **next** state (or the terminal outcome)  
- Update order: **backward** — update the state closest to the terminal first, then propagate.

**Why backward?** Each updated value is more reliable than the old one, so updating from the end lets earlier states benefit from freshly improved downstream information.

---

### Step-by-Step Example — Tic-Tac-Toe (Lec 1 & Lec 2)

**Setup:** State chain $s_1 \to s_2 \to s_3 \to s_4$. Terminal state $s_4 = 1$ (win) or $0$ (loss). $\alpha = 0.82$.

---

#### 📘 Revision PDF Example — 3 Games

| Game | V(s₁) | V(s₂) | V(s₃) | Outcome |
|------|-------|-------|-------|---------|
| Start | 0.5 | — | — | — |
| Game 1 | 0.5 | 0.4 | 0.7 | Win (1) |
| Game 2 | carry forward from Game 1 | 0.6 | 0.5 | Loss (0) |
| Game 3 | carry forward from Game 2 | 0.5 | 0.6 | Win (1) |

**Game 1 — Win ($s_4 = 1$)**

Initial: $V(s_1)=0.5,\ V(s_2)=0.4,\ V(s_3)=0.7,\ V(s_4)=1$

**Step 1:** Update $V(s_3)$ using $V(s_4)$:
$$V(s_3) = 0.7 + 0.82 \times [1 - 0.7] = 0.7 + 0.246 = \boxed{0.946}$$

**Step 2:** Update $V(s_2)$ using the newly updated $V(s_3)$:
$$V(s_2) = 0.4 + 0.82 \times [0.946 - 0.4] = 0.4 + 0.4477 = \boxed{0.8477}$$

**Step 3:** Update $V(s_1)$ using the newly updated $V(s_2)$:
$$V(s_1) = 0.5 + 0.82 \times [0.8477 - 0.5] = 0.5 + 0.285 = \boxed{0.785}$$

**Game 2 — Loss ($s_4 = 0$)**

Carry forward only $V(s_1) = 0.785$. New: $V(s_2)=0.6,\ V(s_3)=0.5$

**Step 1:** Update $V(s_3)$:
$$V(s_3) = 0.5 + 0.82 \times [0 - 0.5] = 0.5 - 0.41 = \boxed{0.09}$$

**Step 2:** Update $V(s_2)$:
$$V(s_2) = 0.6 + 0.82 \times [0.09 - 0.6] = 0.6 - 0.4182 = \boxed{0.182}$$

**Step 3:** Update $V(s_1)$:
$$V(s_1) = 0.785 + 0.82 \times [0.182 - 0.785] = 0.785 - 0.4946 = \boxed{0.290}$$

**Game 3 — Win ($s_4 = 1$)**

Carry forward $V(s_1) = 0.290$. New: $V(s_2)=0.5,\ V(s_3)=0.6$

**Step 1:** $V(s_3) = 0.6 + 0.82 \times [1-0.6] = 0.6 + 0.328 = \boxed{0.928}$

**Step 2:** $V(s_2) = 0.5 + 0.82 \times [0.928 - 0.5] = 0.5 + 0.351 = \boxed{0.851}$

**Step 3:** $V(s_1) = 0.290 + 0.82 \times [0.851 - 0.290] = 0.290 + 0.4597 = \boxed{0.750}$

**Convergence Summary:**

| Game | V(s₁) | V(s₂) | V(s₃) | V(s₄) |
|------|-------|-------|-------|-------|
| 1 | 0.785 | 0.848 | 0.946 | 1 |
| 2 | 0.290 | 0.182 | 0.090 | 0 |
| 3 | 0.750 | 0.851 | 0.928 | 1 |

> 💡 V(s₁) fluctuates but will converge to the true win-probability over many games.

---

#### 📗 Lec 1 Slide Example — XO Tic-Tac-Toe Chain (α = 0.82)

**Setup:** X is building toward a win. The game records 4 board states in sequence. The scores shown are the current estimated win-probabilities **before** updating. α = 0.82.

**Game sequence — board states**

Board 1 ($s_1$):
```
- - X
X O O
X O -
```
Score = 0.500

Board 2 ($s_2$):
```
- X X
X O O
X O -
```
Score = 0.544

Board 3 ($s_3$):
```
- X X
X O O
X O O
```
Score = 0.597

Board 4 ($s_4$):
```
X X X
X O O
X O O
```
Score = 1

**Update rule:** As you make each move $s_t \to s_{t+1}$, immediately back up:
$$V(s_t) \leftarrow V(s_t) + \alpha\big[V(s_{t+1}) - V(s_t)\big]$$

Updates happen **forward in time**, each using the original (not yet updated in this step) value of the next state.

**Move 1 — back up $V(s_1)$ using $V(s_2)$:**
$$V(s_1) = 0.500 + 0.82 \times [0.544 - 0.500] = 0.500 + 0.82 \times 0.044 = \boxed{\approx 0.536}$$

**Move 2 — back up $V(s_2)$ using $V(s_3)$:**
$$V(s_2) = 0.544 + 0.82 \times [0.597 - 0.544] = 0.544 + 0.82 \times 0.053 = \boxed{\approx 0.588}$$

**Move 3 — back up $V(s_3)$ using $V(s_4) = 1.0$:**
$$V(s_3) = 0.597 + 0.82 \times [1.000 - 0.597] = 0.597 + 0.82 \times 0.403 = \boxed{\approx 0.927}$$

**$V(s_4) = 1.000$ — terminal state, no update needed.**

**Slide answer:** `0.536035, 0.587764, 0.927944, 1.000000`

> 💡 **Key difference from the 3-game example:** Here updates happen forward as moves are made (online, in-episode). In the 3-game example updates happen backward after the episode ends. Same formula, different timing.

---

## 2. Greedy Action Selection (MID)

**Problem Name:** Pure Greedy Selection (Sample-Average Method)

**Problem Definition:**  
At each step, always choose the action with the **highest current estimated value** $Q(a)$. Update using the sample average.

**Formula/Approach:**
$$Q_{n+1}(a) = \frac{\sum_{i=1}^{n} R_i}{n} \quad \text{(full average)}$$

Or equivalently (incremental):
$$Q_{n+1}(a) = Q_n(a) + \frac{1}{n}\big[R_n - Q_n(a)\big]$$

**Action selection:** $A_t = \arg\max_a Q_t(a)$

---

### Step-by-Step Example — Doctor / 3 Treatments (Lec 2 + Revision PDF)

**Setup:** 3 treatments A1, A2, A3. True success rates: A1=50%, A2=**70%** (best), A3=40%.  
All estimates start at 0. Reward = 1 (success), 0 (failure).

| Patient | Action | Reward | Q(A1) | Q(A2) | Q(A3) |
|---------|--------|--------|-------|-------|-------|
| Start | — | — | 0 | 0 | 0 |
| 1 | A1 (random tie-break) | 1 | 1.0 | 0 | 0 |
| 2 | A1 (highest) | 0 | 0.5 | 0 | 0 |
| 3 | A1 (highest) | 1 | 0.67 | 0 | 0 |
| 4 | A1 (highest) | 0 | 0.5 | 0 | 0 |
| 5 | A1 (highest) | 1 | 0.6 | 0 | 0 |

**Result:** Doctor **never tried A2** — stuck at A1 (50% rate), missed the 70% optimal.

> ⚠️ **Pitfall:** Greedy gets permanently trapped by early impressions. Never recovers without exploration.

---

## 3. ε-Greedy Action Selection (MID)

**Problem Name:** ε-Greedy Selection

**Problem Definition:**  
Same as greedy but with a small probability $\varepsilon$ of choosing a **random** action (exploration).

**Formula/Approach:**
- With probability $1 - \varepsilon$: **EXPLOIT** — choose $\arg\max_a Q_t(a)$
- With probability $\varepsilon$: **EXPLORE** — choose uniformly at random
- **Decision rule:** Generate random number (RN). If RN $> \varepsilon$ → exploit. If RN $\leq \varepsilon$ → explore.

---

### Step-by-Step Example 1 — Doctor ε = 0.1 (Lec 2 + Revision PDF)

**Setup:** Same as above. $\varepsilon = 0.1$.

| Patient | RN | RN vs ε | Decision | Action | Reward | Q(A1) | Q(A2) |
|---------|-----|---------|----------|--------|--------|-------|-------|
| 1 | 0.35 | >0.1 | EXPLOIT | A1 | 1 | 1.0 | 0 |
| 2 | 0.82 | >0.1 | EXPLOIT | A1 | 0 | 0.5 | 0 |
| 3 | **0.05** | **≤0.1** | **EXPLORE ★** | **A2 (random)** | **1** | 0.5 | **1.0** |
| 4 | 0.40 | >0.1 | EXPLOIT | A2 (highest!) | 1 | 0.5 | 1.0 |
| 5 | 0.60 | >0.1 | EXPLOIT | A2 | 0 | 0.5 | 0.67 |

**Result:** One exploration step (Patient 3) discovered A2 was better. Switched to exploiting A2 — the optimal treatment found.

---

### Step-by-Step Example 2 — Non-Stationary, ε = 0.1, α = 1.0 (Lec 3 + Revision PDF)

**Setup:** Q1=2.36, Q2=1.86, Q3=2.80. With $\alpha=1.0$: $Q_\text{new} = R$ (Q becomes the latest reward).

| Step | Q-values Before | RN | Decision | Action | Reward | Q-values After |
|------|----------------|----|----------|--------|--------|----------------|
| 1 | [2.36, 1.86, 2.80] | 0.45 | Exploit | A3 | 3.2 | [2.36, 1.86, **3.2**] |
| 2 | [2.36, 1.86, 3.2] | **0.08** | **Explore** | **A2** | 2.0 | [2.36, **2.0**, 3.2] |
| 3 | [2.36, 2.0, 3.2] | 0.72 | Exploit | A3 | 2.5 | [2.36, 2.0, **2.5**] |
| 4 | [2.36, 2.0, 2.5] | 0.12 | Exploit | A3 | 1.8 | [2.36, 2.0, **1.8**] |
| 5 | [2.36, 2.0, 1.8] | 0.65 | Exploit | **A1** (now highest!) | 2.4 | [**2.4**, 2.0, 1.8] |
| 6 | [2.4, 2.0, 1.8] | **0.03** | **Explore** | A3 | 2.2 | [2.4, 2.0, **2.2**] |

> 💡 **Key insight:** With $\alpha=1.0$ in a non-stationary environment, the agent can shift focus as the world changes — notice the switch from A3 (steps 1, 3, 4) to A1 (step 5).

---

## 4. Stationary Action-Value Estimation (MID)

**Problem Name:** Incremental Q-Update — Stationary Environment ($\frac{1}{N}$ step size)

**Problem Definition:**  
Given an action sequence with rewards in a **stationary** environment, update $Q(a)$ after each application of action $a$ using the sample-average rule.

**Formula/Approach:**
$$Q_\text{new}(a) = Q_\text{old}(a) + \frac{1}{N(a)}\big[R - Q_\text{old}(a)\big]$$
where $N(a)$ is the **updated** count (after this observation).

---

### Step-by-Step Example — 3 Actions, 20 Steps (Lec 3 + Revision PDF)

**Given:**  
- $Q_1 = 0.98,\ N_1 = 7$; $Q_2 = 0.90,\ N_2 = 3$; $Q_3 = 2.77,\ N_3 = 5$
- Action sequence: 1, 3, 3, 2, 1, 3, 3, 2, 2, 3, 1, 1, 3, 3, 3, 2, 1, 2, 2, 3
- Rewards: -0.59, -0.39, -0.79, -0.90, 0.77, -0.16, -1.02, -0.16, 0.89, 0.76, -1.01, -1.26, 0.24, 0.03, -1.61, -1.34, -0.12, 1.82, -0.42, -1.39

**Action 1 Updates:**

| Seq. Step | R | Calculation | New Q1 | New N1 |
|-----------|---|-------------|--------|--------|
| Start | — | — | 0.9800 | 7 |
| 1 | -0.59 | $0.98 + \frac{1}{8}(-0.59-0.98)$ | **0.7838** | 8 |
| 5 | 0.77 | $0.7838 + \frac{1}{9}(0.77-0.7838)$ | **0.7822** | 9 |
| 11 | -1.01 | $0.7822 + \frac{1}{10}(-1.01-0.7822)$ | **0.6030** | 10 |
| 12 | -1.26 | $0.6030 + \frac{1}{11}(-1.26-0.6030)$ | **0.4337** | 11 |
| 17 | -0.12 | $0.4337 + \frac{1}{12}(-0.12-0.4337)$ | **0.3875** | 12 |

**Action 3 Updates (partial):**

| Seq. Step | R | Calculation | New Q3 | New N3 |
|-----------|---|-------------|--------|--------|
| Start | — | — | 2.7700 | 5 |
| 2 | -0.39 | $2.77 + \frac{1}{6}(-0.39-2.77)$ | **2.2433** | 6 |
| 3 | -0.79 | $2.2433 + \frac{1}{7}(-0.79-2.2433)$ | **1.8100** | 7 |
| 6 | -0.16 | $1.81 + \frac{1}{8}(-0.16-1.81)$ | **1.5638** | 8 |
| 7 | -1.02 | $1.5638 + \frac{1}{9}(-1.02-1.5638)$ | **1.2767** | 9 |

**Full 20-step combined table (selected rows):**

| Step | Action | Reward | Q1 | Q2 | Q3 |
|------|--------|--------|----|----|----|
| 0 | — | — | 0.9800 | 0.9000 | 2.7700 |
| 1 | 1 | -0.59 | 0.7838 | 0.9000 | 2.7700 |
| 2 | 3 | -0.39 | 0.7838 | 0.9000 | 2.2433 |
| 4 | 2 | -0.90 | 0.7838 | 0.4500 | 1.8105 |
| 10 | 3 | 0.76 | 0.7822 | 0.4217 | 1.2254 |
| 20 | 3 | -1.39 | 0.3875 | 0.2878 | 0.6801 |

---

## 5. Non-Stationary Action-Value Estimation

**Problem Name:** Q-Update — Non-Stationary Environment (Constant $\alpha$)

**Problem Definition:**  
When the environment changes over time, use a **constant** learning rate $\alpha$ so recent rewards always matter more.

**Formula/Approach:**
$$Q_\text{new}(a) = Q_\text{old}(a) + \alpha\big[R - Q_\text{old}(a)\big]$$

**Special case $\alpha = 1.0$:** $Q_\text{new}(a) = R$ — Q simply becomes the latest reward (ignores all history).

---

### Step-by-Step Example — α = 1.0 (Lec 3 + Revision PDF)

**Given:**  
- $Q_1=2.36,\ Q_2=1.86,\ Q_3=2.80$, $\alpha=1.0$
- Action sequence: 2, 2, 2, 3, 2, 3, 3, 1, 1, 1, 2, 2, 2, 1, 3, 2, 2, 3, 2, 2
- Rewards: -0.28, -0.71, -0.57, 1.33, -1.57, 1.37, 0.28, -0.84, 2.73, 1.23, ...

| Step | Action | Reward | Q1 | Q2 | Q3 |
|------|--------|--------|----|----|----|
| Initial | — | — | 2.36 | 1.86 | 2.80 |
| 1 | 2 | -0.28 | 2.36 | **-0.28** | 2.80 |
| 2 | 2 | -0.71 | 2.36 | **-0.71** | 2.80 |
| 4 | 3 | 1.33 | 2.36 | -0.57 | **1.33** |
| 8 | 1 | -0.84 | **-0.84** | -1.57 | 0.28 |
| 9 | 1 | 2.73 | **2.73** | -1.57 | 0.28 |
| 18 | 3 | 2.12 | 0.06 | -0.23 | **2.12** |
| 20 | 2 | -2.27 | 0.06 | **-2.27** | 2.12 |

> 💡 With $\alpha=1.0$ every action's Q simply becomes the most recent reward. It reacts instantly to change but forgets the past entirely.

---

## 6. Softmax Action Selection (MID)

**Problem Name:** Softmax (Boltzmann) Action Selection

**Problem Definition:**  
Convert Q-values into a probability distribution using the Boltzmann formula with temperature $\tau$. Select action by sampling from this distribution.

**Formula/Approach:**
$$P(a) = \frac{e^{Q(a)/\tau}}{\sum_{b} e^{Q(b)/\tau}}$$

- Large $\tau$ → flat distribution (more exploration)  
- Small $\tau$ → sharp distribution (more greedy)

**4-Step Algorithm:**
1. Compute $Q(a)/\tau$ for each action
2. Exponentiate: $e^{Q(a)/\tau}$
3. Normalise (divide by sum) → probabilities
4. Build cumulative intervals, sample with random number

---

### Step-by-Step Example — τ = 0.5 (Lec 3 + Revision PDF)

**Initial Q-values:** $Q(A1)=Q(A2)=Q(A3)=0$ → equal probabilities = 1/3 each. Sampling selects **A1, Reward=1**.

After step 0: $Q(A1)=1.0,\ Q(A2)=0,\ Q(A3)=0$

**Patient 1:**

$$\frac{Q(A1)}{\tau} = \frac{1.0}{0.5} = 2.0 \quad \Rightarrow \quad e^{2.0} = 7.389$$
$$\frac{Q(A2)}{\tau} = \frac{Q(A3)}{\tau} = 0.0 \quad \Rightarrow \quad e^{0} = 1.000$$

Sum $= 7.389 + 1 + 1 = 9.389$

$$P(A1) = 7.389/9.389 = \mathbf{78.7\%} \quad P(A2) = P(A3) = 1/9.389 = \mathbf{10.7\%}$$

Cumulative intervals: A1 → [0, 0.787], A2 → [0.787, 0.894], A3 → [0.894, 1.0]

Sampling lands on A1. Reward = 0. $Q(A1) = (1+0)/2 = 0.5$

**Patient 2:** $Q(A1)=0.5 \Rightarrow e^{0.5/0.5}=e^1=2.718$, Sum=4.718

$$P(A1) = 57.6\%,\ P(A2) = P(A3) = 21.2\%$$

Sampling lands on **A2**. Reward = 1. $Q(A2) = 1.0$

**Patient 3:** $Q(A1)=0.5,\ Q(A2)=1.0,\ Q(A3)=0$

$$e^{1.0}=2.718,\ e^{2.0}=7.389,\ e^{0}=1.0,\quad \text{Sum}=11.107$$

$$P(A1)=24.5\%,\ \mathbf{P(A2)=66.5\%},\ P(A3)=9.0\%$$

How to pick: Build intervals → A1: [0, 0.245], A2: [0.245, 0.910], A3: [0.910, 1.0]  
If RN = 0.52 → falls in A2's range → **Choose A2**

**Summary Table:**

| Step | Q(A1) | Q(A2) | Q(A3) | P(A1) | P(A2) | P(A3) | Chosen | Reward |
|------|-------|-------|-------|-------|-------|-------|--------|--------|
| 0 (init) | 0.00 | 0.00 | 0.00 | 33% | 33% | 33% | A1 | 1 |
| 1 | 1.00 | 0.00 | 0.00 | 78.7% | 10.7% | 10.7% | A1 | 0 |
| 2 | 0.50 | 0.00 | 0.00 | 57.6% | 21.2% | 21.2% | A2 | 1 |
| 3 | 0.50 | 1.00 | 0.00 | 24.5% | **66.5%** | 9.0% | A2 | 1 |
| 4 | 0.50 | 1.00 | 0.00 | 24.5% | 66.5% | 9.0% | A2 | 1 |

---

## 7. Upper-Confidence-Bound (UCB) Selection

**Problem Name:** UCB Action Selection

**Problem Definition:**  
Select actions that balance high estimated reward (exploitation) with low observation counts (exploration). Targets actions that are either performing well **or** are highly uncertain.

**Formula/Approach:**
$$A_t = \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N(a)}} \right]$$

- $Q_t(a)$: estimated reward (exploitation term)  
- $c$: confidence parameter  
- $\sqrt{\ln t / N(a)}$: uncertainty bonus — large when $N(a)$ is small

---

### Step-by-Step Example — c=2, t=10 (Lec 3 + Revision PDF)

| Action | Q(a) | N(a) | Bonus = $2\sqrt{\ln 10 / N}$ | Total Score | Decision |
|--------|------|------|-------------------------------|-------------|----------|
| A1 | 5.0 | 100 | $2\sqrt{2.30/100} = 0.30$ | 5.30 | Not chosen |
| A2 | 4.8 | **2** | $2\sqrt{2.30/2} = \mathbf{2.14}$ | **6.94** | **CHOSEN ★** |
| A3 | 4.7 | 3 | $2\sqrt{2.30/3} = 1.75$ | 6.45 | Not chosen |

> 💡 Even though A1 has the highest raw Q (5.0), A2's uncertainty bonus (tried only twice!) makes its total score (6.94) the highest. This is **directed** exploration, not random.

---

## 8. Gradient Bandit / Action Preferences (MID)

**Problem Name:** Action Preference (Gradient Bandit) Update

**Problem Definition:**  
Instead of estimating reward values, maintain a **preference score** $H(a)$ for each action. Higher preference → higher selection probability. Learn preferences using stochastic gradient ascent.

**Formula/Approach:**

Selection probability (Softmax over preferences):
$$\pi(a) = \frac{e^{H(a)}}{\sum_b e^{H(b)}}$$

Update rules after observing reward $R$ and baseline $\bar{R}$:
$$H_\text{new}(\text{chosen}) = H_\text{old}(\text{chosen}) + \alpha (R - \bar{R})(1 - \pi(\text{chosen}))$$
$$H_\text{new}(\text{other}) = H_\text{old}(\text{other}) - \alpha (R - \bar{R}) \cdot \pi(\text{other})$$

Baseline update (running average):
$$\bar{R}_\text{new} = \bar{R}_\text{old} + \frac{1}{n}(R - \bar{R}_\text{old})$$

**Logic:**
- If $R > \bar{R}$ (good surprise): increase chosen action's preference, decrease others
- If $R < \bar{R}$ (bad surprise): decrease chosen action's preference, increase others

---

### Step-by-Step Example 1 — 4 Actions (Lec 3 + Revision PDF)

**Given:**  
$H=[1.26, 0.24, 0.24, 0.22]$, $n=10$, $\bar{R}=0.49$, $\alpha=0.97$  
Actions: 2, 3, 4, 4, 4, 1 | Rewards: 0.31, -0.14, 2.19, -1.49, -1.32, 1.30

**Initial probabilities:**
$$e^{1.26}=3.525,\ e^{0.24}=1.271,\ e^{0.24}=1.271,\ e^{0.22}=1.246,\quad \text{Sum}=7.313$$
$$\pi_1=0.482,\ \pi_2=0.174,\ \pi_3=0.174,\ \pi_4=0.170$$

**Step 1:** Action 2 taken, $R=0.31$

Update baseline ($n \to 11$):
$$\bar{R}_\text{new} = 0.49 + \frac{1}{11}(0.31-0.49) = 0.49 - 0.0164 = 0.4736$$

Surprise factor: $\alpha(R-\bar{R}) = 0.97 \times (0.31-0.49) = -0.1746$

Update $H_2$ (chosen):
$$H_2 = 0.24 + (-0.1746)(1-0.174) = 0.24 - 0.1442 = \mathbf{0.096}$$

Update $H_1$ (not chosen):
$$H_1 = 1.26 - (-0.1746)(0.482) = 1.26 + 0.0842 = \mathbf{1.344}$$

Update $H_3$: $0.24 - (-0.1746)(0.174) = 0.24 + 0.0304 = \mathbf{0.270}$

Update $H_4$: $0.22 - (-0.1746)(0.170) = 0.22 + 0.0297 = \mathbf{0.250}$

**Full Summary Table:**

| Step | Action | Reward | $\bar{R}$ (After) | π₁ | π₂ | π₃ | π₄ | H₁ | H₂ | H₃ | H₄ |
|------|--------|--------|--------------------|----|----|----|----|----|----|----|----|
| 0 | — | — | 0.490 | 0.482 | 0.174 | 0.174 | 0.170 | 1.260 | 0.240 | 0.240 | 0.220 |
| 1 | 2 | 0.31 | 0.474 | — | — | — | — | 1.344 | 0.096 | 0.270 | 0.250 |
| 2 | 3 | -0.14 | 0.423 | 0.509 | 0.146 | 0.174 | 0.170 | 1.647 | 0.183 | **-0.221** | 0.351 |
| 3 | 4 | 2.19 | 0.558 | 0.603 | 0.139 | 0.093 | 0.165 | 0.613 | -0.056 | -0.381 | **1.783** |
| 4 | 4 | -1.49 | 0.412 | 0.196 | 0.100 | 0.073 | 0.631 | 1.003 | 0.143 | -0.236 | 1.049 |
| 5 | 4 | -1.32 | 0.297 | 0.362 | 0.153 | 0.105 | 0.379 | 1.611 | 0.400 | -0.059 | 0.006 |
| 6 | 1 | 1.30 | 0.359 | 0.593 | 0.177 | 0.112 | 0.119 | **2.007** | 0.228 | -0.168 | -0.110 |

---

### Step-by-Step Example 2 — Medical Treatment (2 Actions, Revision PDF)

**Given:** $H(A)=H(B)=0$, $\bar{R}=0$, $\alpha=0.1$  
Initial: $\pi(A)=\pi(B)=50\%$

**Patient 1:** Treatment A selected, $R = +10$

Surprise: $R - \bar{R} = 10 - 0 = +10$

$$H(A) = 0 + 0.1 \times 10 \times (1-0.50) = +0.50$$
$$H(B) = 0 - 0.1 \times 10 \times 0.50 = -0.50$$

Baseline: $\bar{R} = 0 + \frac{1}{1}(10-0) = 10$

New probabilities: $e^{0.5}=1.649,\ e^{-0.5}=0.607,\ \text{Sum}=2.256$
$$\pi(A) = 73\%,\quad \pi(B) = 27\%$$

**Patient 2:** Treatment B selected, $R = -2$

Surprise: $R - \bar{R} = -2 - 10 = -12$ ← very bad!

$$H(B) = -0.50 + 0.1 \times (-12) \times (1-0.269) = -0.50 - 0.8772 = \mathbf{-1.38}$$
$$H(A) = +0.50 - 0.1 \times (-12) \times 0.731 = +0.50 + 0.8772 = \mathbf{+1.38}$$

Baseline: $\bar{R} = 10 + \frac{1}{2}(-2-10) = 4$

New probabilities: $\pi(A) = 94\%,\ \pi(B) = 6\%$

| Patient | Action | Reward | H(A) | H(B) | π(A) | π(B) | R̄ |
|---------|--------|--------|------|------|------|------|---|
| Start | — | — | 0.00 | 0.00 | 50% | 50% | 0 |
| 1 | A | +10 | **+0.50** | **-0.50** | 73% | 27% | 10 |
| 2 | B | -2 | **+1.38** | **-1.38** | **94%** | **6%** | 4 |

---

## 9. MRP / Bellman Equation — State Value & Transition Probability (MID)

**Problem Name:** Bellman Equation — State Value Calculation

**Problem Definition:**  
Given a Markov Reward Process, use the Bellman equation to compute the value of a state as: immediate reward + discounted expected value of next states.

**Formula:**
$$V(s) = R(s) + \gamma \sum_{s'} P(s, s') \cdot V(s')$$

Or in matrix form: $\mathbf{v} = \mathbf{R} + \gamma \mathbf{P}\mathbf{v}$

---

### Step-by-Step Example 1 — State Value from Diagram (Lec 5)

![Example 1 Diagram](Example-1.png)

**Problem:** Calculate the value $V(s)$ of the state circled in red.

**From the diagram:** The red-circled state has:
- Reward: $R = -2$ (for being in / transitioning through this state)
- Transitions to:
  - State with value **10** with probability **0.6**
  - State with value **0.8** with probability **0.4**

Applying the Bellman equation ($\gamma = 1$ absorbed):
$$V(s) = R + P_1 \cdot V(s_1') + P_2 \cdot V(s_2')$$
$$V(s) = -2 + 0.6 \times 10 + 0.4 \times 0.8$$
$$V(s) = -2 + 6 + 0.32 = \boxed{4.32 \approx 4.3}$$

> The slide shows the value **4.3** already on the node, and the red annotation confirms: $4.3 = -2 + 0.6 \times 10 + 0.4 \times 0.8$

---

### Step-by-Step Example 2 — State Value from Transition Matrix (Lec 5)

![Example 2 Matrix](Example-2.png)

**Problem:** Given $\gamma = 0.13$, calculate $V(s_5)$ using the Bellman equation.

**Transition matrix P (5 states, row = from state, column = to state):**

| From \ To | s₁ | s₂ | s₃ | s₄ | s₅ |
|-----------|-----|-----|-----|-----|-----|
| s₁ | 0.00 | 0.00 | 0.42 | 0.00 | 0.58 |
| s₂ | 0.00 | 0.37 | 0.00 | 0.25 | 0.38 |
| s₃ | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s₄ | 0.00 | 0.12 | 0.00 | 0.88 | 0.00 |
| **s₅** | **0.00** | **0.00** | **0.00** | **0.26** | **0.74** |

**Rewards per state:** $R = [-1,\ 13,\ -3,\ 8,\ 12]$

**Current state values:** $V = [39,\ 6,\ -3,\ 5,\ 35]$

**Compute $V(s_5)$ — use row 5 of P:**
$$V(s_5) = R(s_5) + \gamma \sum_{j=1}^{5} P(s_5, s_j) \cdot V(s_j)$$
$$= 12 + 0.13 \times \big[0.00 \times 39 + 0.00 \times 6 + 0.00 \times (-3) + 0.26 \times 5 + 0.74 \times 35\big]$$

**Inner sum:**
$$0.26 \times 5 = 1.30$$
$$0.74 \times 35 = 25.90$$
$$\text{Sum} = 1.30 + 25.90 = 27.20$$

**Final calculation:**
$$V(s_5) = 12 + 0.13 \times 27.20 = 12 + 3.536 = \boxed{15.536}$$

---

### Step-by-Step Example 3 — Transition Probability (Lec 5)

**Formula:** $P_{i,j} = \sum_a \pi(a|s_i) \cdot P(s_j | s_i, a)$

$$S_{1,1} = 0.19 \times 0.41 + 0.30 \times 0 + 0.19 \times 0 + 0 \times 0.0 + 0.32 \times 0.92 = \mathbf{0.38}$$
$$S_{3,2} = 0.06 \times 0.56 + 0.24 \times 0.78 + 0.31 \times 0.77 + 0 \times 0.17 + 0.39 \times 0.33 = \mathbf{0.588}$$

---

## 10. Policy Evaluation (Iterative DP) — Gridworld

**Problem Name:** Iterative Policy Evaluation

**Problem Definition:**  
Given a deterministic environment, equi-probable random policy, and reward $-1$ for all transitions, compute state values by repeated Bellman backups until convergence.

**Formula/Approach:**
$$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\big[R + \gamma V_k(s')\big]$$

For equi-probable policy with 4 actions ($\pi=0.25$ each), deterministic transitions, $\gamma=1$, $R=-1$:
$$V_{k+1}(s) = 0.25 \sum_{s' \in \text{neighbors}} \big[-1 + V_k(s')\big]$$

---

### Step-by-Step Example — 14-State Gridworld (Lec 6)

**Setup:** 14 non-terminal states, 4 actions (up/down/left/right), $\gamma=1$, $R=-1$, equi-probable policy.

**K=0 (initialization):** All non-terminal states $V(s)=0$, terminal states $=0$.

**K=1** — Each state gets:
$$V_1(s) = 0.25 \times [(-1+0)+(-1+0)+(-1+0)+(-1+0)] = 0.25 \times (-4) = -1$$

(Edge states: actions hitting the wall stay in the same state, so still average out to $-1$)

**K=2** — Terminal neighbours now contribute. States adjacent to terminal get pulled toward 0:
$$V_2(\text{state 1}) \approx -1.7$$

> The value propagates outward from the terminal state. After many sweeps, the policy improvement step reveals the shortest path to the terminal.

---

## 11. Policy Improvement — Q-value Computation

**Problem Name:** Policy Improvement via Q-value Calculation

**Problem Definition:**  
Given a value function $V_\pi$ and discount factor $\gamma$, compute the Q-value of each action in a state and pick the best one (greedy improvement).

**Formula/Approach:**
$$Q(s, a) = \sum_{s'} P(s'|s,a)\big[R(s,a,s') + \gamma V_\pi(s')\big]$$

Pick the improved action: $\pi'(s) = \arg\max_a Q(s,a)$

---

### Step-by-Step Example — State S4, γ = 0.61 (Lec 6)

**Given:** State S4 with 4 possible actions. Discount factor $\gamma = 0.61$.

$$Q(s_4, a_1) = \sum_{s'} P(s'|s_4,a_1)[R + 0.61 \cdot V(s')]$$
$$Q(s_4, a_2) = \ldots$$
$$Q(s_4, a_3) = \ldots$$
$$Q(s_4, a_4) = \ldots \quad \text{(highest value)}$$

**Result:** Since Action 4 has the highest (least negative) Q-value, the policy for state S4 improves to: **choose Action 4**.

> 💡 Policy improvement always moves toward higher Q. If the greedy action matches the current policy, we've converged.

---

## 12. Asynchronous DP — In-Place Updates

**Problem Name:** Asynchronous (In-Place) Dynamic Programming

**Problem Definition:**  
Unlike synchronous DP (update all states simultaneously using old values), asynchronous DP updates states in any order using the **latest available** values immediately.

**Formula/Approach:** Same Bellman backup as Policy Evaluation, but applied state-by-state in any order, with each update immediately overwriting the old value.

---

### Step-by-Step Example — States 14→13 (Lec 6)

**Setup:** All $V(s)=-5.0$ (random init), terminal $V=0$, $\gamma=1$, $R=-1$.

**Update State 14 first** (adjacent to terminal):
$$V(14) = -5.0 + 0.69 \times [-1 + 1 \times V(\text{terminal}) - (-5.0)] = -5.0 + \ldots = \mathbf{-2.0}$$

**Immediately update State 13** using the freshly updated V(14):
$$V(13) = -5.0 + 0.69 \times [-1 + 1 \times (-2.0) - (-5.0)] = -5.0 + \ldots = \mathbf{-2.0}$$

| Method | State 13 result |
|--------|----------------|
| Synchronous | Stays at -6.0 until next full sweep |
| **Asynchronous** | **Improves to -2.0 immediately** |

---

## 13. Monte Carlo Prediction (State-Value Update)

**Problem Name:** Monte Carlo State-Value Estimation

**Problem Definition:**  
After observing a complete episode, compute the **return** $G_t$ (discounted sum of future rewards) for each visited state, then update the state's value estimate using the MC update rule.

**Formula/Approach:**

Return from time $t$:
$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{T-t-1} R_T$$

**Non-stationary update (constant $\alpha$):**
$$V(S_t) \leftarrow V(S_t) + \alpha\big[G_t - V(S_t)\big]$$

**Stationary update (1/N):**
$$V(S_t) \leftarrow V(S_t) + \frac{1}{N(S_t)+1}\big[G_t - V(S_t)\big]$$

---

### Step-by-Step Example 1 — Non-Stationary (Lec 7)

**Episode:** States [4, 1, 6, 6, 5, 2, 3] — terminal state is 3  
**Rewards per state (1→6):** [5.00, 1.00, 0.00, -1.00, 0.00, -2.00]  
**Current V(s) per state (1→6):** [-17.00, -7.00, 0.00, -7.00, -12.00, -27.00]  
$\gamma=0.17$, $\alpha=0.03$

**Compute Returns $G_t$ (backward from terminal):**

The reward for each visited state comes from the reward of the state it's in.  
State 3 (terminal) → Reward = 0 (no future)  
State 2 → Reward for state 2 = 1.00, $G = 1.00 \times \gamma^0 + 0 = 0$  
Backward computation gives:

| t | State | $G_t$ | Old V | Calculation | New V |
|---|-------|--------|-------|-------------|-------|
| 0 | 4 | 4.6030 | -7.00 | $-7.00 + 0.03(4.603 - (-7))$ | **-6.652** |
| 1 | 1 | -2.3351 | -17.00 | $-17.00 + 0.03(-2.335 - (-17))$ | **-16.560** |
| 2 | 6 | -1.9711 | -27.00 | $-27.00 + 0.03(-1.971 - (-27))$ | **-26.249** |
| 3 | 6 | 0.1700 | -26.249 | $-26.249 + 0.03(0.17 - (-26.249))$ | **-25.457** |
| 4 | 5 | 1.0000 | -12.00 | $-12.00 + 0.03(1.0 - (-12))$ | **-11.610** |
| 5 | 2 | 0.0000 | -7.00 | $-7.00 + 0.03(0 - (-7))$ | **-6.790** |

---

### Step-by-Step Example 2 — Stationary (Lec 7)

**Episode:** States [5, 1, 5, 1, 5, 2, 1, 3] — terminal state = 3  
**Rewards per state (1→5):** [5.00, 4.00, 0.00, 5.00, -11.00]  
**Current V per state (1→5):** [-30, -18, 0, -3, -10]  
**N per state:** [17, 17, 10, 16, 19], $\gamma=0.45$

| t | State | $G_t$ | New N | Old V | New V |
|---|-------|--------|-------|-------|-------|
| 0 | 5 | 0.3164 | 20 | -10.00 | **-9.484** |
| 1 | 1 | -10.408 | 18 | -30.00 | **-28.912** |
| 2 | 5 | 1.3156 | 21 | -9.484 | **-8.970** |
| 3 | 1 | -8.1875 | 19 | -28.912 | **-27.821** |
| 4 | 5 | 6.2500 | 22 | -8.970 | **-8.278** |
| 5 | 2 | 5.0000 | 18 | -18.00 | **-16.722** |
| 6 | 1 | 0.0000 | 20 | -27.821 | **-26.430** |

---

## 14. Monte Carlo Action-Value (Q-table) Update

**Problem Name:** Monte Carlo Q(s,a) Update (First-Visit)

**Problem Definition:**  
Estimate Q(s,a) values by averaging returns from state-action pairs visited in an episode. Used when no environment model exists (can't compute $V$ alone to derive policy).

**Formula/Approach:**

For each $(s_t, a_t)$ pair visited (first visit only):
$$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$$
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\big[G_t - Q(s_t, a_t)\big]$$

The **greedy path** afterwards: starting from each state, always pick the action with the highest Q-value.

---

### Step-by-Step Example — Grid Navigation (Lec 7)

**Episode:** 9→8→7→8→7→4→Terminal  
**Rewards:** -1, -1, -1, -1, -1, 0  
**Actions:** a1=Right, a2=Left, a3=Down, a4=Up  
$\alpha=0.79$, $\gamma=0.06$

**Step summary:**

| t | $S_t$ | $S_{t+1}$ | $R_{t+1}$ | Action |
|---|-------|-----------|-----------|--------|
| 0 | 9 | 8 | -1.00 | a4 (up) |
| 1 | 8 | 7 | -1.00 | a4 (up) |
| 2 | 7 | 8 | -1.00 | a3 (down) |
| 3 | 8 | 7 | -1.00 | a4 (up) |
| 4 | 7 | 4 | -1.00 | a2 (left) |
| 5 | 4 | Terminal | 0.00 | — |

**Compute Returns (from end backward):**
$$G_5 = 0,\quad G_4 = -1 + 0.06 \times 0 = -1.0$$
$$G_3 = -1 + 0.06 \times (-1) = -1.06,\quad G_2 = -1 + 0.06 \times (-1.06) = -1.0636$$
$$G_1 = -1 + 0.06 \times (-1.0636) = -1.0638,\quad G_0 = -1 + 0.06 \times (-1.0638) = -1.0638$$

**Update Q(8, a4):**
$$Q(8, a4)_\text{new} = Q(8, a4)_\text{old} + 0.79 \times [G_1 - Q(8, a4)_\text{old}]$$
$$= 0 + 0.79 \times [-1.0638 - 0] = \mathbf{-0.840}$$

---

## 15. Monte Carlo Control with ε-Soft Policy

**Problem Name:** MC Control with ε-Greedy Path Selection

**Problem Definition:**  
After updating the Q-table using MC, follow an ε-soft policy: with probability $1-\varepsilon$ pick the greedy action, with probability $\varepsilon$ explore randomly.

**Formula/Approach:**
- Greedy action: $a^* = \arg\max_a Q(s, a)$
- $P(a^*) = (1-\varepsilon) + \varepsilon/|A|$
- $P(\text{other}) = \varepsilon/|A|$
- Decision: if RN $> \varepsilon$ → exploit, else → explore randomly

---

### Step-by-Step Example — City Navigation (Lec 7)

**After MC update**, Q-matrix is updated. ε = 0.1.

**Greedy path selection (exploit only):**

| City | Best Q-value | Greedy Action → Next City |
|------|-------------|--------------------------|
| C1 | 5.02 (C4) | → C4 |
| C4 | -8.34 (C8) | → C8 |
| C8 | 12.10 (C10) | → C10 |
| C10 | -56.60 (C11) | → C11 |

**Greedy path:** $C1 \to C4 \to C8 \to C10 \to C11$

**ε-greedy path** (ε=0.1, using random numbers):

| Current City | RN | Decision | Action | Next City |
|-------------|-----|----------|--------|-----------|
| C1 | 0.42 | Greedy (>0.1) | Max Q = C4 | **C4** |
| C4 | 0.87 | Greedy (>0.1) | Max Q = C8 | **C8** |
| C8 | **0.04** | **EXPLORE (<0.1)** | Random → **C9** | C9 |
| C9 | 0.61 | Greedy (>0.1) | Max Q = C11 | **C11** |

**ε-greedy path:** $C1 \to C4 \to C8 \to C9 \to C11$ (different from greedy due to exploration!)

---

## 16. TD(0) State-Value Update

**Problem Name:** TD(0) — One-Step Temporal Difference

**Problem Definition:**  
Update the value of the **current state** using just the **next step's reward and the next state's value** — no need to wait for the episode to end. This is the simplest TD method.

**Formula/Approach:**
$$V(S_t) \leftarrow V(S_t) + \alpha\big[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\big]$$

The term $R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is the **TD error** (the "surprise").

| Method | Target used for update |
|--------|----------------------|
| MC | Actual return $G_t$ (whole episode) |
| **TD(0)** | **$R_{t+1} + \gamma V(S_{t+1})$ (one step ahead)** |
| DP | $\sum P \cdot [R + \gamma V]$ (all transitions) |

---

### Step-by-Step Example (Lec 8)

**Setup:** 4 states. $\alpha=0.69$, $\gamma=0.09$. Transitions: S4→S1, S1→S3, S3→S2(terminal).

| State | Initial V | Reward | Bootstrap V (next) | Calculation | New V |
|-------|-----------|--------|--------------------|-------------|-------|
| S4 | -26.00 | -8.00 | V(S1)=-21.00 | $-26 + 0.69[-8 + 0.09\times(-21) - (-26)]$ | **-14.88** |
| S1 | -21.00 | -13.00 | V(S3)=-15.00 | $-21 + 0.69[-13 + 0.09\times(-15) - (-21)]$ | **-16.41** |
| S3 | -15.00 | -1.00 | V(S2)=0.00 | $-15 + 0.69[-1 + 0.09\times 0 - (-15)]$ | **-5.34** |
| S2 | 0.00 | — | Terminal | No update | **0.00** |

**Detailed calculation for S4:**
$$V(S4) = -26 + 0.69 \times [-8 + 0.09 \times (-21) - (-26)]$$
$$= -26 + 0.69 \times [-8 - 1.89 + 26]$$
$$= -26 + 0.69 \times 16.11 = -26 + 11.12 = \mathbf{-14.88}$$

---

## 17. SARSA — On-Policy TD Control

**Problem Name:** SARSA (State-Action-Reward-State-Action)

**Problem Definition:**  
On-policy TD control. Updates Q(s,a) after each step using the **actual next action** chosen by the current policy (not the greedy best).

**Formula/Approach:**
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\big[R_{t+1} + \gamma \cdot Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\big]$$

The key difference from Q-learning: $Q(S_{t+1}, A_{t+1})$ uses the action the agent **actually takes next** (from policy $\pi$), not the maximum possible Q-value.

---

### Step-by-Step Example — Grid Navigation (Lec 8)

**Episode:** 6→3→6→9 | Rewards: -1, 1, -1, 0  
$\alpha=0.66$, $\gamma=0.71$

**Step 1: S=6, A=a2(left), R=-1, S'=3, A'=a1(right)**
$$Q(6,2) = 0 + 0.66 \times [-1 + 0.71 \times Q(3,1) - 0]$$
$$= 0 + 0.66 \times [-1 + 0.71 \times 0] = 0.66 \times (-1) = \mathbf{-0.66}$$

**Step 2: S=3, A=a1(right), R=1, S'=6, A'=a1(right)**
$$Q(3,1) = 0 + 0.66 \times [1 + 0.71 \times Q(6,1) - 0]$$
$$= 0.66 \times [1 + 0.71 \times 0.74] = 0.66 \times 1.5254 = \mathbf{-0.31}$$

Wait — reward is +1 but we get −0.31? Let me show correctly:
$$Q(3,1) = 0 + 0.66 \times [-1 + 0.71 \times 0.74 - 0]$$
$$= 0.66 \times [-1 + 0.5254] = 0.66 \times (-0.4746) = \mathbf{-0.31}$$

**Step 3: S=6, A=a1(right), R=-1, S'=9, A'=?**
$$Q(6,1) = 0.74 + 0.66 \times [-1 + 0 - 0.74]$$
$$= 0.74 + 0.66 \times (-1.74) = 0.74 - 1.1484 = \mathbf{-0.41}$$

---

### Example 2 — SARSA in City Navigation (Lec 8)

**Episode:** Cities 1→2→4→6→8→11  
$\alpha=0.43$, $\gamma=0.75$

| Transition | Q_old | R | Q_next | Calculation | Q_new |
|------------|-------|---|--------|-------------|-------|
| C1→C2 | -2.77 | -6.95 | Q(2,4)=15.07 | $-2.77+0.43(-6.95+0.75\times15.07+2.77)$ | **0.29** |
| C2→C4 | 15.07 | (reward) | Q(4,6)=-0.26 | $15.07+0.43(\ldots)$ | **-5.21** |
| C4→C6 | -0.26 | (reward) | Q(6,8)=3.55 | $-0.26+0.43(\ldots)$ | **-28.78** |
| C6→C8 | 3.55 | (reward) | Q(8,11)=10.69 | $3.55+0.43(\ldots)$ | **-30.33** |
| C8→C11 | 10.69 | (reward) | 0 (terminal) | $10.69+0.43(\ldots)$ | **-40.36** |

**Greedy path after update:** 1 → 3 → 6 → 10 → 11

---

## 18. Q-Learning — Off-Policy TD Control

**Problem Name:** Q-Learning

**Problem Definition:**  
Off-policy TD control. Updates Q(s,a) using the **maximum Q-value of the next state** (greedy action), regardless of what action the policy actually takes. Directly approximates $q^*$ (optimal).

**Formula/Approach:**
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\big[R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)\big]$$

| | SARSA | Q-Learning |
|-|-------|-----------|
| Update uses | Actual next action $A_{t+1}$ | $\max_{a'} Q(S_{t+1}, a')$ |
| Policy type | On-policy | Off-policy |
| Learns | $q_\pi$ (current policy) | $q^*$ (optimal) |

---

### Step-by-Step Example — City Navigation (Lec 8)

**Episode:** C1→C3→C6→C7→C9→C10  
$\alpha=0.06$, $\gamma=0.42$

**Step 1: C1→C3**
$$Q(1,3)_\text{old} = 10.99,\quad R=-27.15,\quad \max_{a'} Q(C3,\cdot) = Q(C3,C6)=9.00$$
$$Q(1,3) = 10.99 + 0.06 \times [-27.15 + 0.42 \times 9.00 - 10.99]$$
$$= 10.99 + 0.06 \times [-27.15 + 3.78 - 10.99]$$
$$= 10.99 + 0.06 \times (-34.36) = 10.99 - 2.06 = \mathbf{8.93}$$

**Full update table:**

| Transition | Q_old | R | max Q(s') | Updated Q |
|------------|-------|---|-----------|-----------|
| C1→C3 | 10.99 | -27.15 | 9.00 (C6) | **8.92** |
| C3→C6 | 9.00 | -28.38 | 6.20 (C7) | **6.91** |
| C6→C7 | 6.20 | -65.31 | 2.05 (C8) | **1.96** |
| C7→C9 | -6.05 | -44.19 | -16.61 (C10) | **-8.76** |
| C9→C10 | -16.61 | -99.51 | 0 (terminal) | **-21.58** |

**Final Greedy Path** (after update):
1. C1: max(Q_C2=-10.88, Q_C3=**8.93**, Q_C4=-17.24, Q_C5=4.82) → **C3**
2. C3: max(Q_C4=1.85, Q_C5=-19.95, Q_C6=**6.91**, Q_C7=1.88) → **C6**
3. C6: max(Q_C7=**1.96**, Q_C8=-2.72, ...) → **C7**
4. C7: max(Q_C8=**2.05**, Q_C9=-8.76, ...) → **C8**
5. C8: max(Q_C9=-10.14, Q_C10=**-1.86**) → **C10**

**Final path:** $C1 \to C3 \to C6 \to C7 \to C8 \to C10$

---

## 19. n-Step TD Return Calculation

**Problem Name:** n-Step TD Return

**Problem Definition:**  
A middle ground between TD(0) (1-step) and Monte Carlo (full episode). Look ahead exactly $n$ steps, collect real rewards, then bootstrap with the value estimate at step $n$.

**Formula/Approach:**

n-step return:
$$G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$$

State value update:
$$V(S_t) \leftarrow V(S_t) + \alpha\big[G_{t:t+n} - V(S_t)\big]$$

| n | Method | Bias | Variance |
|---|--------|------|----------|
| 1 | TD(0) | High | Low |
| $\infty$ | Monte Carlo | Zero | High |
| **2–5** | **n-step TD** | **Medium** | **Medium** |

---

### Step-by-Step Example 1 — 2-Step Return (Lec 9)

**Episode:** States [1,5,7,2,1,2,4] — terminal=last  
**Rewards:** -3, -1, 2, -3, -3, -3, 0  
**V values:** -4, -1, -4, 0, -5, -5, -1  
$\gamma=0.95$, $n=2$, compute from $t=0$

$$G_{0:2} = R_1 + \gamma R_2 + \gamma^2 V(S_2)$$
$$= (-3.00) + 0.95 \times (-1.00) + (0.95)^2 \times V(\text{state at step 2})$$

State at step 2 = state 7, $V(7) = -1$:
$$G_{0:2} = -3.00 + 0.95 \times (-1.00) + 0.9025 \times (-1.00)$$
$$= -3.00 - 0.95 - 0.9025 = \mathbf{-4.85}$$

---

### Step-by-Step Example 2 — 5-Step Return (Lec 9)

**Episode:** States [1,5,1,4,4,2] — terminal  
**Rewards:** -1, -1, 1, -3, -3, 0  
**V values:** -4, 0, -5, -2, -1  
$\gamma=0.22$, $n=5$, $t=0$

$$G_{0:5} = (-1) + 0.22(-1) + 0.22^2(1) + 0.22^3(-3) + 0.22^4(-3) + 0.22^5(0)$$
$$= -1 - 0.22 + 0.0484 - 0.032 - 0.0070 + 0$$
$$= \mathbf{-1.21} \approx -1.3$$

> Note: At step 5 we reach the terminal state (reward=0), so the bootstrap term is 0.

---

## 20. Complex Return TD(λ)

**Problem Name:** Complex Return / TD(λ)

**Problem Definition:**  
Average multiple n-step returns using geometric weights controlled by $\lambda$. TD(λ) spans the spectrum: $\lambda=0$ gives TD(0), $\lambda=1$ gives Monte Carlo.

**Formula/Approach:**

$\lambda$-return:
$$G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_{t:t+n}$$

Weights:
- 1-step: weight $(1-\lambda)$
- 2-step: weight $(1-\lambda)\lambda$  
- 3-step: weight $(1-\lambda)\lambda^2$  
- etc.

---

### Step-by-Step Example — λ=0.52 (Lec 9)

**Episode:** States [3, 7, 5, 1] → Terminal  
**Rewards:** $R_1=-1.00,\ R_2=-3.00,\ R_3=-3.00,\ R_4=0.00$ (terminal)  
**State values:** $V(1)=0,\ V(2)=-1,\ V(3)=-2,\ V(4)=-1,\ V(5)=-1,\ V(6)=-1,\ V(7)=-1$  
$\gamma=0.15$, $\lambda=0.52$

**Episode path:** $S_0=3,\ S_1=7,\ S_2=5,\ S_3=1$ → Terminal

---

**Step 1 — Compute the individual n-step returns from $S_0=3$:**

**1-step return:**
$$G_{0:1} = R_1 + \gamma\, V(S_1) = -1.00 + 0.15 \times V(7)$$
$$= -1.00 + 0.15 \times (-1.00) = -1.00 - 0.15 = \mathbf{-1.15}$$

**2-step return:**
$$G_{0:2} = R_1 + \gamma R_2 + \gamma^2\, V(S_2)$$
$$= -1.00 + 0.15 \times (-3.00) + (0.15)^2 \times V(5)$$
$$= -1.00 - 0.45 + 0.0225 \times (-1.00)$$
$$= -1.00 - 0.45 - 0.045 = \mathbf{-1.495}$$

**3-step return** (episode ends — $V(S_3) = V(1) = 0$):
$$G_{0:3} = R_1 + \gamma R_2 + \gamma^2 R_3 + \gamma^3\, V(S_3)$$
$$= -1.00 + 0.15 \times (-3.00) + (0.15)^2 \times (-3.00) + (0.15)^3 \times 0$$
$$= -1.00 - 0.45 - 0.0675 + 0 = \mathbf{-1.5175}$$

> The series is **finite** here — the episode ends at step 3, so no further n-step returns exist.

---

**Step 2 — Compute the λ-return using geometric weights:**

$$G_0^\lambda = (1-\lambda)\big[\lambda^0 G_{0:1} + \lambda^1 G_{0:2} + \lambda^2 G_{0:3}\big]$$

| n | Weight $(1-\lambda)\lambda^{n-1}$ | $G_{0:n}$ | Contribution |
|---|----------------------------------|-----------|---------------|
| 1 | $(0.48)(1) = 0.48$ | $-1.15$ | $0.48 \times (-1.15) = -0.5520$ |
| 2 | $(0.48)(0.52) = 0.2496$ | $-1.495$ | $0.2496 \times (-1.495) = -0.3732$ |
| 3 | $(0.48)(0.52)^2 = 0.1298$ | $-1.5175$ | $0.1298 \times (-1.5175) = -0.1970$ |

$$G_0^\lambda = -0.5520 + (-0.3732) + (-0.1970) = \boxed{\approx -1.12}$$

---

## 21. DQN — TD Target & Loss Calculation

**Problem Name:** Deep Q-Network (DQN) TD Target and Loss

**Problem Definition:**  
In DQN, instead of updating a Q-table cell, the agent trains a neural network by minimising the **Mean Squared Error** between its current Q-value prediction and the **TD Target** (a better estimate). This is the only numeric computation in Lec 10.

**Formula/Approach:**

$$\text{TD Target} = r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a';\,\theta^-)$$

$$\text{TD Error} = \text{TD Target} - Q(s_t, a_t;\,\theta)$$

$$\text{Loss} = (\text{TD Error})^2$$

Where:
- $Q(s_t, a_t;\,\theta)$ = the **online network's** current predicted Q-value for the action taken
- $Q(s_{t+1}, a';\,\theta^-)$ = the **target network's** Q-values for the next state (used to stabilise training)
- $\gamma$ = discount factor
- $r_{t+1}$ = immediate reward received

**Key components unique to DQN vs plain Q-learning:**

| Component | Purpose |
|-----------|---------|
| **Neural Network** | Approximates Q(s,a) for large/continuous state spaces (e.g. raw pixels) |
| **Experience Replay** | Stores $(s, a, r, s')$ tuples in a buffer; trains on random mini-batches to break temporal correlation |
| **Target Network** | A frozen copy of the network updated periodically — prevents the "chasing a moving target" instability |

---

### Step-by-Step Example — DQN Loss (Lec 10)

**Scenario:** You are in state $s_t$. Your network predicts the current move is worth **10 points**.

| Value | Number |
|-------|--------|
| Current Q-estimate: $Q(s_t, a_t)$ | **10** |
| Immediate reward received: $r_{t+1}$ | **2** |
| Best Q in next state: $\max_{a'} Q(s_{t+1}, a')$ | **9** |
| (Assume $\gamma = 1$ for simplicity) | |

**Step 1:** Compute the TD Target:
$$\text{TD Target} = r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') = 2 + 1 \times 9 = \mathbf{11}$$

**Step 2:** Compute the TD Error:
$$\text{Error} = \text{TD Target} - Q(s_t, a_t) = 11 - 10 = \mathbf{1}$$

**Step 3:** Compute the Loss:
$$\text{Loss} = (\text{Error})^2 = (1)^2 = \mathbf{1}$$

**Step 4:** Backpropagation — the agent uses this Loss=1 to nudge the network weights $\theta$ so that next time it sees the same state, its prediction moves closer to 11.

> 💡 **Why MSE?** Squaring the error penalises large mistakes more heavily, and makes the loss differentiable for gradient descent.

> ⚠️ **Pitfall:** Without a **Target Network**, both the prediction ($Q$) and the target ($\text{TD Target}$) shift every step — like trying to hit a moving bullseye. The Target Network freezes the target for a number of steps, making training stable.

---

## Quick-Reference Comparison Table

| Algorithm | Model? | Updates | Bootstraps? | On/Off Policy |
|-----------|--------|---------|-------------|---------------|
| DP (Policy Iteration) | ✅ Yes | After sweep | ✅ Yes | — |
| MC | ❌ No | After episode | ❌ No | On or Off |
| **TD(0)** | ❌ No | After each step | ✅ Yes | On (SARSA) / Off (Q) |
| SARSA | ❌ No | After each step | ✅ Yes | **On-policy** |
| Q-Learning | ❌ No | After each step | ✅ Yes | **Off-policy** |
| n-step TD | ❌ No | After n steps | ✅ Yes | On or Off |

---

## Formulas Quick Sheet

$$\text{TD Update: } V(s) \leftarrow V(s) + \alpha[V(s') - V(s)]$$

$$\text{Incremental Q: } Q_{n+1} = Q_n + \frac{1}{n}[R_n - Q_n]$$

$$\text{Constant } \alpha\text{: } Q_\text{new} = Q_\text{old} + \alpha[R - Q_\text{old}]$$

$$\text{Softmax: } P(a) = \frac{e^{Q(a)/\tau}}{\sum_b e^{Q(b)/\tau}}$$

$$\text{UCB: } A_t = \arg\max_a \left[Q_t(a) + c\sqrt{\frac{\ln t}{N(a)}}\right]$$

$$\text{Gradient: } H_\text{chosen} \mathrel{+}= \alpha(R-\bar{R})(1-\pi),\quad H_\text{other} \mathrel{-}= \alpha(R-\bar{R})\pi$$

$$\text{Bellman: } V_\pi(s) = \sum_a \pi(a|s)\sum_{s'} P(s'|s,a)[R + \gamma V_\pi(s')]$$

$$\text{SARSA: } Q(s,a) \leftarrow Q(s,a) + \alpha[R + \gamma Q(s',a') - Q(s,a)]$$

$$\text{Q-Learning: } Q(s,a) \leftarrow Q(s,a) + \alpha[R + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

$$\text{n-step: } G_{t:t+n} = \sum_{k=1}^{n} \gamma^{k-1} R_{t+k} + \gamma^n V(S_{t+n})$$

$$\text{TD}(\lambda)\text{: } G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1} G_{t:t+n}$$

$$\text{DQN Loss: } \mathcal{L} = \left(r + \gamma \max_{a'} Q(s', a';\,\theta^-) - Q(s, a;\,\theta)\right)^2$$
