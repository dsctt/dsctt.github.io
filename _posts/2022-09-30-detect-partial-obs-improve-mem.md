---
title: 'Detecting partial observability in decision processes and improving value with memory'
excerpt: 'We explore a method aimed at reliably detecting aliasing in POMDPs and using this signal to search for memory functions that allow for finding higher performing policies--all without previous knowledge of the state-space.'
date: 2022-09-30
permalink: /blog/2022/09/detect-partial-obs-improve-mem/
author_profile: false
tags:
  - reinforcement learning
---

The content of this article comes from work done at the [Intelligent Robot Lab](http://irl.cs.brown.edu/) at Brown University during the summer of 2022 alongside [Cam Allen](https://camallen.net/), [Saket Tiwari](https://tsaket.github.io/), [Sam Lobel](https://samlobel.github.io/),  [Anita de Mello Koch](https://www.linkedin.com/in/anita-de-mello-koch-6423b486), [Aaron Kirtland](https://atkirtland.github.io/), [David Tao](http://taodav.cc/), [Omer Gottesman](https://cs.brown.edu/people/faculty/ogottesm/), [Michael Littman](https://www.littmania.com/), and [George Konidaris](https://cs.brown.edu/people/gdk/). It is a semi-technical exposition intended for an audience that has some familiarity with the fundamental concepts in Reinforcement Learning. We explore a method aimed at reliably detecting aliasing in POMDPs and using this signal to search for memory functions that allow for finding higher performing policies--all without previous knowledge of the state-space.

---

1\. Decision processes and the problem of partial observability
---

Reinforcement learning (RL) is concerned with studying how an agent that is interacting with the environment around it can learn to take the best action in any given state of the environment. This interaction is conceptualized as a loop where the agent observes the current state of the environment, reasons internally about which action to take based on that observation, and then executes the action which then changes the state of the environment for the next time around (visualized in Figure 1). The objective is to learn the optimal policy which contains, for each observation, the action selection that will lead to the highest expected cumulative reward at the end of the interaction loop, which occurs when the environment reaches a terminal state.

![](images/figure1.png)
**Figure 1:** *Agent-Environment interaction loop. The environment state is observed by the agent which then takes an action that affects the environment. At the time of receiving the next observation, the agent also receives a reward associated with the most recent action.*

Our focus here is on how the nature of the observations changes the dynamics of the learning process. If the observations that the agent sees perfectly distinguish each state of the environment from every other state, then the environment is considered to be **fully observable**. If instead the same observation can be seen when in different states (that is, different states actually look the same to the agent), then this introduces **partial observability** into the situation, making the problem of finding the best action to take at each step much harder to solve.

As a simple example, consider an agent that is navigating a series of rooms that are identical except for the color of their walls. To get to each successive room, the agent must choose to go through either the upper or lower passageway. Each passageway gets the agent to the next room but one of them may be associated with a desirable outcome which we’ll formalize as +1 reward. The agent must learn which passageway to take in each room to accumulate the highest possible reward. 

One can imagine a circumstance where there are two states that look the same but the best action to take in each of the two states is different. In this case, the two states are **aliased** due to them emitting the same observation and, furthermore, it is impossible to learn and execute those separate best actions based on the observation alone. **A method to detect and disambiguate these aliased states in any given environment could allow the agent to ultimately learn to take the best action after all. To accomplish this, we will look into adding more information to the observation: memory of past events in the trajectory.**


![](images/figure2.png)
**Figure 2:** *Environment A. An MDP with four states, each of which looks different from the others, so observing the color is equivalent to being told exactly which state the agent is in as far as decision-making is concerned. At each state, one of two actions can be taken: UP or DOWN. The discount factor we will use is $\gamma = 0.5$. Adapted from a visualization by Cam Allen.*

We will begin by adding some formalism to our simple example, starting with a fully observable case as a baseline. The sequential decision-making problem in our example can be modeled as a Markov Decision Process (MDP) as shown in Figure 2, named Environment A. There, the colors indicate the observation seen at that state (which we can think of as the wall colors in the simple example). In Environment A, the observations uniquely identify the underlying states, so the environment is fully observable. The optimal policy can be attained by only looking at the *current* color and selecting an action based on that. In this environment, any policy, $\pi$, where $\pi(red)=DOWN$ and $\pi(green)=UP$ is optimal and will result in the agent obtaining the maximum possible reward of +2.

Now, contrast this with Environment B in Figure 3 which shows a new environment that is very similar to Environment A except that there are two states that look the same (i.e. have the same wall color) so we consider Environment B to be a Partially Observable MDP (POMDP). Both environments are Markovian in the sense that the underlying dynamics depend only on information from the current timestep. However, in Environment B, the best memory-based policy can do better than the best memoryless policy due to the aliasing resulting from the observation function behavior. In particular, to get the maximum possible reward in this environment (+2), one needs to keep track of history to know what to do upon seeing a red observation. Is the agent at the *first* red-looking state? Then take DOWN and get +1. Or the *second* one? Then take UP and get +1.  

![](images/figure3.png)
**Figure 3:** *Environment B. A POMDP with four states. Similar to Environment A, but here, two of the states look exactly the same. Observing that it's in a red room does not help an agent determine whether it should go UP or DOWN. $\gamma = 0.5$. Adapted from a visualization by Cam Allen.*

There is, however, an issue that must be addressed before the agent is able to attempt to answer these questions. Imagine that as the agent, you are dropped into Environment B having never seen it before, starting in the first red room. You look at the wall color, select a passageway according to your policy, and show up in a blue room. After which, you select another action and show up once again in a red room. Your only sense being of the current wall color, you then select your action for red and conclude your adventure, landing in the purple room. The second red room was treated exactly the same way as the first red room. An RL agent will happily learn the best policy it can here with standard algorithms (only achieving a maximum expected cumulative reward of +1) without noticing that the two red encounters were actually different states and, importantly, had different best actions that if always selected would result in an even better performing policy. However, as we saw in this walkthrough, from the agent's perspective, it was entirely possible that, instead, there actually was one red state and that it just stochastically sends you to blue or purple.

*How can a learning agent detect that there is in fact aliasing in an environment and that it's not e.g. looping back to the same red state in a fully observable environment?*

2\. Policy evaluation and the $\lambda$-discrepancy
---

We've seen that the agent cannot detect aliasing by looking at the observations alone as it executes the policy. We need something else that the agent can latch on to as it traverses the POMDP--something that it can use as a signal of aliasing so that it can eventually learn a policy that achieves the +2 return. 

Often in RL, an intermediate step in finding the optimal policy is estimating the value of states, $V(s)$, or state-action pairs, $Q(s,a)$. In the POMDP case, these (and the policy) are considered to be over observations: $V(o)$ and $Q(o,a)$. The values are useful in iterating toward the optimal policy because we would like to find the policy that achieves the highest expected return, and this expected return is what the values represent. 

One method to estimate these values is to run the policy in question on the MDP and record what the average discounted returns are from each observation-action pair in a Monte-Carlo (MC) fashion. This is a natural algorithm to apply because it uses the real rewards seen over the course of the interaction to generate an expectation of what return will be achieved from any point in the MDP. This is particularly important for accuracy in the POMDP setting because, while the agent is only making decisions based on observations, its actions on the environment still move it within the underlying state space, and MC records the reward outcomes of this movement without any bias (i.e. without intermediate estimations) and applies them to the observation seen at that state. In this way, it doesn't get tripped up by the aliasing.

On the other hand, another typical policy evaluation option *can* get tripped up as a result of the issue due to bias that MC avoids. The degree of bias can be tuned using the [TD($\lambda$)](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf) algorithm wherein a $\lambda$ value of 1 provides results equivalent to MC and lower values of $\lambda$ dial up the bias. So at the other extreme, in TD(0), the value of an observation is estimated to be the immediate real reward upon leaving a state plus the current estimated value of the next observation (a.k.a. bootstrapping off of the next observation). This means that, crucially, TD(0) during bootstrapping will have been erroneously sharing values of any states that emitted the same observation. What does this look like in our Environment B? When TD(0) estimates the value of the actions at the blue observation, it is bootstrapping off of the value of the red observation that is seen at the next state, but two different states are contributing to this value! 

The assumption of sub-problem composability underlying the TD(0) update breaks due to the aliasing in a partially observable environment like our example. So while TD(1) and TD(0) converge to the same result in a fully observable environment since states have unique observations, this may not be the case in a partially observable one. The different behaviors of the TD($\lambda$) algorithm with different values of $\lambda$ when there is aliasing may lead to the signal we are looking for. In particular, we have something, in the TD(1)-TD(0) value estimation discrepancy, that seems to be directly affected by the presence of aliasing. We can verify that this discrepancy--which we'll call the **$\lambda$-discrepancy**--is present in our example and then we'll need to consider the general reliability of this as a signal.

The hypothesis that we have now is that **running TD($\lambda$) with $\lambda=1$ and $\lambda=0$ over the observation space of a given environment and considering their discrepancy provides a signal of memory-improvable partial observability**. Namely, if the $Q$ values of any observation-action pair estimated by TD(0) and TD(1) differ--in other words, there is a non-zero **$\lambda$-discrepancy**--then at least one observation must have been emitted by multiple underlying states and, furthermore, there is some memory that can be implemented to help achieve our overall goal of higher returns. In Environment B, that memory would answer the questions from Section 1 as to which red state the agent is in.

To explore the theory here, we can analytically determine the values that these two variations of the TD algorithm will converge to in expectation given the full set of quantities comprising the environment model. This means we know the finite set of possible states $S$, the finite set of possible actions $A$, the finite set of possible observations $Z$, and all of the quantities that define the POMDP: transition function ($T$), reward function ($R$), observation function ($\phi$), starting state distribution ($P_0$), and discount factor($\gamma$). First, we calculate the values of the states of the underlying MDP using the following system of linear equations:

$$ \forall s, V_\pi(s) = \sum_{s'} \sum_{a} T(s'|s,a) \pi(a|s) (R(s,a,s') + \gamma V_\pi(s')) $$

where $a \in A$, $s \in S$ is the current state, $s' \in S$ is the next state, and $\pi$ is the policy under evaluation. Let's make the policy "always take UP":

| Obs    | P(UP) | P(DOWN) |
| ---    | ------- | -------- |
| red    | 1       | 0        |  
| blue   | 1       | 0        |  
| green  | 1       | 0        |
| purple | 1       | 0        |

Solving this system in our example environments (which have equivalent underlying MDPs) gives:

| Q(state, action) | Value |
| ---- | ---- |
| Q(S0, UP) | 0.25 |
| Q(S0, DOWN) | 1.25 |
| Q(S1, UP) | 0.5 |
| Q(S1, DOWN) | 0.5 |
| Q(S2, UP) | 1.0 |
| Q(S2, DOWN) | 0.0 |
| Q(S3, UP) | 0.0 |
| Q(S3, DOWN) | 0.0 |

Now we will calculate the expected Q values for TD(1) when it is run in the observation space of Environment B. Since TD(1) averages the discounted sum of rewards after each visit to an observation, we can estimate the result by aggregating the values of the underlying states according to their expected “contribution” to the value of each observation. This contribution is a weighting by $P_\pi(s\|o)$. Intuitively, we are considering how much each state is expected to have been visited as compared to each other state that could have emitted the observation in question and weighting the values of commonly visited states more. We begin by calculating the expected visitation count, or **occupancy**, for each state using the system of equations:

$$ \forall s, C_\pi(s) = P_0(s) + \sum_{s^{-1}} \sum_{a} C_\pi(s^{-1}) \gamma T(s|a,s^{-1}) \pi(a|s^{-1}) $$

where $s^{-1} \in S$ is the previous state. For each observation, we can then multiply this occupancy vector (one entry per state) by $P(o\|s)$ which is a column from $\phi$. Normalizing the result gives us $P_\pi(s\|o)$ which we can then use to weight and sum the values of each state to get the TD(1) values for the observation:

```python
# pseudocode to aggregate state values into TD(1) observation values given
# - underlying mdp_q_vals (vector of size S)
# - occupancy (vector of size S)
# - phi (SxZ matrix)
for ob in num_obs:
  p_π_of_o_given_s = phi[:, ob] # a vector containing prob for each state of emitting "ob"
  w = occupancy * p_π_of_o_given_s
  p_π_of_s_given_o = w / w.sum() # normalize

  weighted_q_vals = (mdp_q_vals * p_π_of_s_given_o).sum(1) # weight and sum, keeping the actions dimension
  final_q_vals[:, ob] = weighted_q_vals # save the q vals for current ob
```

The analogous analytical procedure for TD(0) involves aggregating the transition and reward functions to create a new "effective TD(0) model" as shown in Figure 4. The effective model is the (possibly incorrect) MDP that TD(0) sees as it is running and updating according to its bootstrapping mechanism. This MDP might seem familiar as it also represents the fully observable possibility from the walkthrough at the end of Section 2, where, from the agent's perspective, there might have been one red state that stochastically sends you to either blue or purple.

![](images/figure4.png)
**Figure 4:** *The effective TD(0) model for Environment B. After red, with either action, it appears to TD(0) as though there is a 50/50 chance of going to blue or purple. Adapted from a visualization by Cam Allen.*

The idea is to aggregate state-state relationships within the POMDP dynamics into observation-observation relationships. Aggregating $T$ and $R$ requires the same $P_\pi(s\|o)$ object to weight contributions but we are operating in higher dimensions now. Both the current and next observations must be considered so the following pseudocode shows how it would work with two `for` loops and maintaining the actions dimension: 


```python
# pseudocode to aggregate state values into TD(0) observation values given
# - occupancy (vector of size S)
# - phi (SxZ matrix)
# - T (AxSxS tensor)
# - R (AxSxS tensor)
for curr_ob in num_obs:
  # Get p_π_of_s_given_o just like before in TD(1)
  p_π_of_o_given_s = phi[:, curr_ob] # a vector containing prob for each state of emitting "curr_ob"
  w = occupancy * p_π_of_o_given_s
  p_π_of_s_given_o = w / w.sum() # normalize

  for next_ob in num_obs:
    p_π_of_op_given_sp = phi[:, next_ob] # a vector containing prob for each (next) state of emitting "next_ob" 

    # Weight contributions of each state to the curr_ob -> next_ob transition.
    # The first multiplication weights for transitions from curr_ob and the second weights for transitions to next_ob.
    T_contributions = T * p_π_of_s_given_o * p_π_of_op_given_sp 

    T_obs_obs[:, curr_ob, next_ob] = T_contributions.sum(2).sum(1) # sum along the two state dimensions, keeping the actions dimension

    # Calculate how much of each state -> state transition reward will apply to the curr_ob -> next_ob transition reward
    # by considering how likely that state -> state transition is to have been the one that underlied the curr_ob -> next_ob transition
    # and weighting the reward for that state -> state transition by this likelihood
    R_contributions = R * T_contributions / T_obs_obs # divide to normalize

    R_obs_obs[:, curr_ob, next_ob] = R_contributions.sum(2).sum(1) # sum along the two state dimensions, keeping the actions dimension

```

Once acquired, this effective model is then solved using the same procedure as the underlying MDP. The results for our Environment B are shown in the following table:

| Q(observation, action) | TD(0) | TD(1) | $\lambda$-discrepancy |
| ---- | ---- | ---- | ---- |
| Q(red, UP) | 0.25 | 0.4 | 0.15 |
| Q(red, DOWN) | 0.85 | 1.0 | 0.15 |
| Q(blue, UP) | 0.125 | 0.5 | 0.375 |
| Q(blue, DOWN) | 0.125 | 0.5 | 0.375 |
| Q(purple, UP) | 0.0 | 0.0 | 0.0 |
| Q(purple, DOWN) | 0.0 | 0.0 | 0.0 |

We can see a discrepancy in the values of the red and blue observations--**our signal of memory-improvable partial observability**. To reiterate, this analytical procedure of calculating the values is useful for exploring the idea, but in practice an agent will have to sample transitions from the environment to estimate these values as it will not have access to all of the quantities used in the calculations.

Another point to note is the role of the policy in this method and a resulting susceptibility of the policy evaluation procedure. Above, the arbitrary policy "always take UP" was chosen. It is possible in some POMDP environments with memory-improvable aliasing for the policy to be adversarially selected such that the environment is effectively Markovian under that policy. This occurs for Environment B for any policy where $\pi(UP \| red)=4/7$ as shown in Figure 5. These are the only such policies for this environment. They can be found analytically using value equations or programmatically by following the negative gradient of the discrepancy with respect to the policy parameters which one can imagine when looking at Figure 5. How significant is this phenomenon for our objective? If we randomly initialize the stochastic policy that we use for our evaluation, it is virtually impossible for it to be one of these adversarial ones. Thus far, empirical analysis supports this notion and suggests that this may be the case for any other environment as well which would mean that these policies always make up an infinitely small proportion of the stochastic policy space.

![](images/figure5.png)
**Figure 5:** *Heatmap of the $\lambda$-discrepancy in the space of policies over the red and blue observations in Environment B. The axis values are the probabilities of taking UP in that observation. There is 0 discrepancy when the probability of taking UP in red is 4/7=~0.5714. The discrepancy being constant along the horizontal axis indicates that the policy for blue does not affect the discrepancy, which is due to it having no effect on rewards in this particular environment--the rewards and transitions for both actions from this observation are equivalent.*

*How can we use this signal to pinpoint what exactly should be remembered by the agent to improve returns?*

3\. Improving value with memory
---
Histories can differentiate aliased states. Even if the states have the same observation outputs, the way one reaches them will be different. Otherwise, in this framework, if two states look the same, and are reachable through the exact same series of transitions, then they are, in effect, the exact same state. We could have the agent remember every observation it has seen thus far and condition the policy on that history instead of just the current observation. So at the first red state it will have "red" going into the policy, and at the second red state the input will be "red+blue+red". This method would solve our problem as the policy can be learned to include $\pi(red)=DOWN$ and $\pi(red+blue+red)=UP$, but it is apparent that this would be a cumbersome procedure. The possible combinations of observations is infinite in theory. 

We would like to come up with compact memory functions instead of trying to remember the entirety of the history so we need a way to represent and search the space of memory functions. Here, we consider those representable as a finite state machine (FSM) outputting a certain number of bits. In our Environment B, it turns out that the maximum improvement is achievable with just one memory bit and there are multiple one-bit memory functions that work. Figure 6 shows one such memory function. Basically, the function keeps track of whether the most recent observation was blue. If it was, the memory transitions to 1. If it wasn't, it resets to 0. 

![](images/figure6.png)
**Figure 6:** *A memory function that resolves the partial observability in Environment B. We assume that memory always starts at 0. The agent makes its first decision with augmented observation red+M0. After the first red observation, the memory function stays at 0 and the agent sees blue+M0. Then, after the blue observation, memory changes to 1 giving red+M1 in the next step. Adapted from a visualization by Cam Allen.*

This memory function actually fully resolves the partial observability in the environment which may not always be possible across all POMDPs (in those cases some value improvement may still be possible). The agent can now observe both the observation from the environment and the state of its memory function when making decisions. During an episode, instead of `red -> blue -> red -> purple` it will now see `red+M0 -> blue+M0 -> red+M1 -> purple+M0`. Importantly, the two states that looked the same (both red), now look different (`red+M0` vs. `red+M1`). The agent can now learn about them separately and make unique decisions!

An analytical verification can be done just like before by first taking the cross-product of the underlying MDP and the memory FSM to create the MDP+memory space that the agent will be moving through as shown in Figure 7.

![](images/figure7.png)
**Figure 7:** *Cross-product space of Environment B from Figure 3 and the memory function FSM from Figure 6. Adding 1 bit of memory doubles the size of the state space. The agent can observe the M value along with the color.*

All of the same procedures apply to this MDP to get the expected TD(0) and TD(1) values:

| Q(observation, action) | TD(0) | TD(1) | $\lambda$-discrepancy |
| ---- | ---- | ---- | ---- |
| Q(red+M0, UP) | 0.25 | 0.25 | 0.0 |
| Q(red+M0, DOWN) | 1.25 | 1.25 | 0.0 |
| Q(red+M1, UP) | 1.0 | 1.0 | 0.0 |
| Q(red+M1, DOWN) | 0.0 | 0.0 | 0.0 |
| Q(blue+M0, UP) | 0.5 | 0.5 | 0.0 |
| Q(blue+M0, DOWN) | 0.5 | 0.5 | 0.0 |
| Q(blue+M1, UP) | 0.0 | 0.0 | 0.0 |
| Q(blue+M1, DOWN) | 0.0 | 0.0 | 0.0 |
| Q(purple+M0, UP) | 0.0 | 0.0 | 0.0 |
| Q(purple+M0, DOWN) | 0.0 | 0.0 | 0.0 |
| Q(purple+M1, UP) | 0.0 | 0.0 | 0.0 |
| Q(purple+M1, DOWN) | 0.0 | 0.0 | 0.0 |

Since adding 1 bit of memory doubled the size of the state and observation spaces, we need a method of aggregating the observation-with-memory values back into observation-without-memory values if we want to compare to the pre-memory values. To do so, we use another weighting and summing procedure where e.g. the values for `red+M0` and `red+M1` are combined based on their relative occupancies. After this final aggregation, we get:

| Q(state, action) | TD(0) | TD(1) | $\lambda$-discrepancy |
| ---- | ---- | ---- | ---- |
| Q(red, UP) | 0.4 | 0.4 | 0.0 |
| Q(red, DOWN) | 1.0 | 1.0 | 0.0 |
| Q(blue, UP) | 0.5 | 0.5 | 0.0 |
| Q(blue, DOWN) | 0.5 | 0.5 | 0.0 |
| Q(purple, UP) | 0.0 | 0.0 | 0.0 |
| Q(purple, DOWN) | 0.0 | 0.0 | 0.0 |

**There is no longer a discrepancy which supports the hypothesis that the discrepancy is a signal of memory-improvable partial observability.** Reaching this point would mean that adding more bits of memory would not allow the agent to get any higher returns than it currently can. This result tracks with our understanding that disambiguating the two states with red observations is enough for an agent to then learn the policy achieving maximum possible cumulative reward using an algorithm like Q-learning. This policy would be any one that includes $\pi(DOWN\| red+M0)=1$ and $\pi(UP\| red+M1)=1$.

Finally, how do we go about finding these memory functions? Similarly to how the adversarial policy could be found by descending the gradient of the discrepancy with respect to the policy parameters, perhaps we can do the same with the memory function parameters since the FSM can be represented with a transition tensor. We can check this with our analytical implementation by using an auto-diff library like [jax](https://github.com/google/jax). This means iterating through stochastic transitions, but initial evidence suggests that the value-improving memory functions can indeed be found through this method on an example like ours here. Once again, a final implementation will need to be sample-based to be practical. Putting all of these pieces together into an algorithm that iterates on the memory function in addition to the policy and determining the cases where it will succeed in achieving of our objective is the subject of ongoing work.

---
*Thanks to Cam Allen and David Paulius for feedback on an initial draft of this article.*
