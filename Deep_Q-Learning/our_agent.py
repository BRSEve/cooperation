# our_agent.py
from collections import namedtuple
import numpy as np
import random
import networkx as nx
import DQN
import torch
import math
import json
import torch.nn.functional as F
import os

main_dir = os.path.dirname(os.path.realpath(__file__))
main_dir = main_dir + '/'
with open(main_dir + 'Setting.json') as f:
    setting = json.load(f)

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))
train_times = setting["DQN"]["train_times"]

f = open("experiences", "a")

class QAgent(object):
    def __init__(self, dynetwork):
        self.config = {
            "nodes": dynetwork.num_nodes,
            "epsilon": setting['AGENT']['epsilon'],
            "decay_rate": setting['AGENT']['decay_epsilon_rate'],
            "batch_size": setting['DQN']['memory_batch_size'],
            "gamma": setting['AGENT']['gamma_for_next_q_val'],

            "update_less": setting['DQN']['optimize_per_episode'],
            "sample_memory": setting['AGENT']['use_random_sample_memory'],
            "recent_memory": setting['AGENT']['use_most_recent_memory'],
            "priority_memory": setting['AGENT']['use_priority_memory'],

            "update_epsilon": False,
            "update_models": torch.zeros([1, dynetwork.num_nodes], dtype=torch.bool),
            "entered": 0,
        }
        self.adjacency = dynetwork.adjacency_matrix
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def act(self, neural_network, state, neighbor, sec_penalty=None):
        """
        ε-greedy；利用时对每个邻居动作的 Q 值减去传入的安全惩罚（越安全惩罚越小）。
        """
        if random.uniform(0, 1) < self.config['epsilon']:
            return None if not neighbor else random.choice(neighbor)
        else:
            if not neighbor:
                return None
            with torch.no_grad():
                qvals = neural_network.policy_net(state.float())
                scores = qvals[:, neighbor].clone()
                if sec_penalty is not None and len(neighbor) > 0:
                    pen = torch.tensor(sec_penalty, dtype=scores.dtype, device=scores.device).view(1, -1)
                    scores = scores - pen
                next_step_idx = scores.argmax().item()
                next_step = neighbor[next_step_idx]
                if self.config['update_epsilon']:
                    self.config['epsilon'] = self.config["decay_rate"] * self.config['epsilon']
                    self.config['update_epsilon'] = False
            return next_step

    def learn(self, nn, dqns, current_event, action, reward, next_state):
        if (action is None) or (reward is None):
            return
        if current_event is not None:
            nn.replay_memory.push(current_event, action, next_state, reward)
            f.writelines([f"nn.ID's memory: {[nn.ID]}\n"])
            f.writelines([f"experience--State: {[current_event]}\n"])
            f.writelines([f"experience--action: {[action]}\n"])
            f.writelines([f"experience--next_state: {[next_state]}\n"])
            f.writelines([f"experience--reward: {[reward]}\n"])

        if (self.config["update_models"][:, nn.ID]) & (nn.replay_memory.can_provide_sample(self.config['batch_size'])):
            if self.config['sample_memory']:
                experiences = nn.replay_memory.sample(self.config['batch_size'])
            elif self.config['recent_memory']:
                experiences = nn.replay_memory.take_recent(self.config['batch_size'])
            elif self.config['priority_memory']:
                experiences, experiences_idx = nn.replay_memory.take_priority(self.config['batch_size'])

            states, actions, next_states, rewards = self.extract_tensors(experiences)
            next_states = next_states.to(self.device)
            rewards = rewards.to(self.device)

            current_q_values = self.get_current_QVal(nn.policy_net, states, actions)
            next_q_values = self.get_next_QVal(dqns, next_states, actions)
            target_q_values = (next_q_values * self.config['gamma']) + rewards

            if self.config['priority_memory']:
                nn.replay_memory.update_priorities(experiences_idx, current_q_values, torch.transpose(target_q_values, 0, 1))

            loss = F.mse_loss(current_q_values, torch.transpose(target_q_values, 0, 1))
            nn.optimizer.zero_grad()
            loss.backward()
            nn.optimizer.step()

    def extract_tensors(self, experiences):
        states = torch.cat(tuple(exps[0] for exps in experiences))
        actions = torch.cat(tuple(torch.tensor([exps[1]]) for exps in experiences))
        next_states = torch.cat(tuple(exps[2] for exps in experiences))
        rewards = torch.cat(tuple(torch.tensor([exps[3]]) for exps in experiences))
        return (states, actions, next_states, rewards)

    def get_current_QVal(self, policy_net, states, actions):
        states = states.to(self.device)
        actions = actions.type(torch.int64).to(self.device)
        return policy_net(states.float()).gather(dim=1, index=actions.unsqueeze(-1))

    def get_next_QVal(self, dqns, next_states, actions):
        """
        计算下一状态的目标Q值：
        - 用动作对应的 target_net 评估 next_states；
        - 只在 “未到达目的地” 的样本上取 “邻居可达动作” 的最大Q；
        - 统一使用 PyTorch 的 bool 掩码索引，避免 NumPy 布尔索引导致的类型错误。
        """
        batch_size = next_states.shape[0]
        dest_indices = torch.argmax(next_states, dim=1).detach().cpu().tolist()
        act_indices = actions.detach().cpu().numpy().astype(int)

        all_q = torch.empty(batch_size, self.config['nodes'], device=self.device)
        for i, a_idx in enumerate(act_indices):
            all_q[i] = dqns[a_idx].target_net(next_states[i].float())

        values = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            if int(act_indices[i]) != int(dest_indices[i]):
                adjs_np = (self.adjacency[act_indices[i]] == 1)
                adjs_mask = torch.from_numpy(adjs_np).to(self.device)
                values[i] = torch.max(all_q[i][adjs_mask]).detach()
            else:
                values[i] = 0.0

        return values.view(1, -1)
