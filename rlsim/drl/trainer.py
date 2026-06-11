# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:40:00 2022

@author: 17000
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ase import Atoms
from rgnn.graph.dataset.reaction import ReactionDataset
from rgnn.graph.reaction import ReactionGraph
from rgnn.graph.utils import batch_to
from rgnn.train.trainer import AverageMeter
from torch_geometric.loader import DataLoader

from rlsim.drl.replay_buffer import PrioritizedReplayBuffer


class Trainer:
    def __init__(
            self,
            model,
            # logger,
            q_params: Dict[str, Dict[str, float] | bool | float],
            offline_model= None,
            lr=10**-3,
            train_all=True,
    ):
        self.policy_value_net = model
        # self.logger = logger
        self.target_net = offline_model
        if not train_all:
            for name, param in self.policy_value_net.named_parameters():
                if "reaction_model" in name:
                    param.requires_grad = False
        trainable_params = filter(
            lambda p: p.requires_grad, self.policy_value_net.parameters()
        )
        self.optimizer = optim.Adam(trainable_params, lr=lr)
        self.q_params = q_params
        # self.kT = q_params["temperature"] * 8.617 * 10**-5

    def update(self, replay_buffer: PrioritizedReplayBuffer, mode="dqn", **kwargs):
        if mode == "dqn":
            loss = self.dqn_update(replay_buffer, **kwargs)
        else:
            loss = self.context_bandit_update(replay_buffer, **kwargs)
        return loss
    
    def context_bandit_update(
        self, replay_buffer: PrioritizedReplayBuffer, num_epoch, batch_size=8, device="cuda"
    ):
        self.policy_value_net.to(device)
        self.policy_value_net.train()
        losses = AverageMeter()
        transitions, indices, weights = replay_buffer.sample()
        if not transitions:
            return 0.0

        states = [t["state"] for t in transitions]
        taken_actions = [t["taken_action"] for t in transitions]
        barrier = torch.tensor([t["barrier"] for t in transitions], dtype=torch.float)
        freq = torch.tensor([t["freq"] for t in transitions], dtype=torch.float)
        weights = torch.tensor(weights, dtype=torch.float)

        dataset_list = []
        for i, state in enumerate(states):
            graph_list = convert(state, taken_actions[i])
            for data in graph_list:
                data.q1 = freq[i].unsqueeze(0)
                data.q0 = (-1 * barrier[i]).unsqueeze(0)
                data.buffer_idx = indices[i]
                data.importance_weight = weights[i]
                dataset_list.append(data)
        dataset = ReactionDataset(dataset_list)
        q_dataloader = DataLoader(dataset, batch_size=batch_size)

        for _ in range(num_epoch):
            for batch in q_dataloader:
                batch = batch_to(batch, device)
                output = self.policy_value_net(batch, q_params=self.q_params)
                importance_weight = batch["importance_weight"].to(device)
                q0_error = (batch.q0 - output["q0"]).abs()
                q1_error = (batch.q1 - output["q1"]).abs()
                td_errors = (q0_error + q1_error).detach()
                loss = torch.mean(
                    importance_weight * (q0_error ** 2 + q1_error ** 2)
                )
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(
                    self.policy_value_net.parameters(), 1.0
                )
                self.optimizer.step()
                losses.update(loss.item())
                replay_buffer.update_priorities(
                    batch["buffer_idx"].cpu().numpy(),
                    td_errors.cpu().numpy(),
                )

        return losses.avg

    def get_max_Q(self, model, state, action_space, device="cuda"):
        dataset_list = convert(state, action_space)
        dataset = ReactionDataset(dataset_list)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        q_values_list = []
        with torch.no_grad():
            model.eval()
            model.to(device)
            for batch in dataloader:
                batch = batch_to(batch, device)
                q = model(batch, q_params=self.q_params)["rl_q"]
                q_values_list.append(q.detach())
        del dataset, dataloader
        next_q = torch.cat(q_values_list, dim=0)
        max_q = torch.max(next_q)
        return max_q

    def dqn_update(
        self, replay_buffer: PrioritizedReplayBuffer, gamma, num_epoch, batch_size=8, device="cuda"
    ):
        self.target_net.load_state_dict(self.policy_value_net.state_dict())
        self.target_net.eval()
        self.target_net.to(device)
        self.policy_value_net.train()
        self.policy_value_net.to(device)
        losses = AverageMeter()

        transitions, indices, weights = replay_buffer.sample()
        if not transitions:
            return 0.0

        states = [t["state"] for t in transitions]
        next_states = [t["next_state"] for t in transitions]
        taken_actions = [t["taken_action"] for t in transitions]
        next_aspace = [t["next_aspace"] for t in transitions]
        rewards = torch.tensor([t["reward"] for t in transitions], dtype=torch.float)
        weights = torch.tensor(weights, dtype=torch.float)

        next_Q = torch.zeros(len(next_states))
        for i, state in enumerate(next_states):
            max_q = self.get_max_Q(
                self.target_net, state, next_aspace[i], device=device
            )
            next_Q[i] = max_q
        dataset_list = []
        for i, state in enumerate(states):
            graph_list = convert(state, taken_actions[i])
            for data in graph_list:
                data.rl_q = next_Q[i] * gamma + rewards[i]
                data.buffer_idx = indices[i]
                data.importance_weight = weights[i]
                dataset_list.append(data)
        dataset = ReactionDataset(dataset_list)
        q_dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        for _ in range(num_epoch):
            for batch in q_dataloader:
                batch = batch_to(batch, device)
                q_pred = self.policy_value_net(batch, q_params=self.q_params)[
                    "rl_q"
                ]
                importance_weight = batch["importance_weight"].to(device)
                td_errors = (batch["rl_q"] - q_pred).abs().detach()
                loss = torch.mean(importance_weight * td_errors ** 2)
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(
                    self.policy_value_net.parameters(), max_norm=0.8
                )
                self.optimizer.step()
                losses.update(loss.detach().item())
                replay_buffer.update_priorities(
                    batch["buffer_idx"].cpu().numpy(),
                    td_errors.cpu().numpy(),
                )
                del batch, q_pred, loss
        if device == "cuda":
            torch.cuda.empty_cache()
        del dataset, q_dataloader
        
        return losses.avg


def convert(atoms: Atoms, actions: List[List[float]]) -> List[ReactionGraph]:
    traj_reactant = []
    traj_product = []
    for act in actions:
        traj_reactant.append(atoms)
        final_atoms = atoms.copy()
        final_positions = []
        for i, pos in enumerate(final_atoms.get_positions()):
            if i == act[0]:
                new_pos = pos + act[1:]
                final_positions.append(new_pos)
            else:
                final_positions.append(pos)
        final_atoms.set_positions(final_positions)
        traj_product.append(final_atoms)
    graph_list = []
    for i in range(len(traj_reactant)):
        data = ReactionGraph.from_ase(traj_reactant[i], traj_product[i])
        graph_list.append(data)

    return graph_list
