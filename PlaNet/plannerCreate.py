from math import inf
import torch
from torch import jit


# Model-predictive control planner with cross-entropy method and learned transition model
# class MPCPlanner(jit.ScriptModule):
class MPCPlanner(torch.nn.Module):
  __constants__ = ['action_size', 'planning_horizon', 'optimisation_iters', 'candidates', 'top_candidates', 'min_action', 'max_action']

  def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model, reward_model, min_action=-inf, max_action=inf,num_tools=36):
    super().__init__()
    self.transition_model, self.reward_model = transition_model, reward_model
    self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
    self.planning_horizon = planning_horizon
    self.optimisation_iters = optimisation_iters
    self.candidates, self.top_candidates = candidates, top_candidates
    self.num_tools = num_tools

  # @jit.script_method
  def forward(self, belief, state):
    B, H, Z = belief.size(0), belief.size(1), state.size(1)
    belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
    # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
    cont_action_size = 2
    action_mean, action_std_dev = torch.zeros(self.planning_horizon, B, 1, cont_action_size, device=belief.device), torch.ones(self.planning_horizon, B, 1, cont_action_size, device=belief.device)
    discrete_action_freq = torch.ones(self.planning_horizon, B, 1, self.num_tools)
    for _ in range(self.optimisation_iters):
      # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
      discrete_action_freq_flat = discrete_action_freq.reshape(-1,discrete_action_freq.shape[-1])
      discrete_action = torch.multinomial(discrete_action_freq_flat,self.candidates,replacement=True)
      discrete_action = discrete_action.reshape(self.planning_horizon,B*self.candidates,1).to(action_mean.device)
      actions = (action_mean + action_std_dev * torch.randn(self.planning_horizon, B, self.candidates, cont_action_size, device=action_mean.device)).view(self.planning_horizon, B * self.candidates, cont_action_size)  # Sample actions (time x (batch x candidates) x actions)
      actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
      actions = torch.cat([discrete_action,actions],dim=-1)
      # Sample next states
      beliefs, states, _, _ = self.transition_model(state, actions, belief)
      # Calculate expected returns (technically sum of rewards over planning horizon)
      returns = self.reward_model(beliefs.view(-1, H), states.view(-1, Z)).view(self.planning_horizon, -1).sum(dim=0)
      # Re-fit belief to the K best action sequences
      _, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
      topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)  # Fix indices for unrolled actions
      best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates, 1+cont_action_size)
      # Update belief with new means and standard deviations
      action_mean, action_std_dev = best_actions[...,1:].mean(dim=2, keepdim=True), best_actions[...,1:].std(dim=2, unbiased=False, keepdim=True)
      discrete_action_freq = torch.ones(self.planning_horizon, B, self.num_tools,1)
      discrete_actions = best_actions[...,0]
      histograms = torch.zeros((discrete_actions.shape[0], discrete_actions.shape[1], 1, self.num_tools))
      for i in range(discrete_actions.shape[0]):
        for j in range(discrete_actions.shape[1]):
          histograms[i, j, 0] = torch.histc(discrete_actions[i, j], bins=self.num_tools, min=0, max=self.num_tools)
          histograms[i, j, 0] += torch.ones_like(histograms[i, j, 0])
          histograms[i, j, 0] /= sum(histograms[i, j, 0])

      discrete_action_freq = histograms
    # Return first action mean µ_t
    _,discrete_action = torch.max(discrete_action_freq[0],dim=-1)
    discrete_action = discrete_action.to(action_mean.device)
    cont_action = action_mean[0].squeeze(dim=1).to(action_mean.device)
    action = torch.cat([discrete_action,cont_action],dim= 1)
    # return action_mean[0].squeeze(dim=1)
    return action