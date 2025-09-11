import json
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import copy
import dynetwork
import gym
from gym import error
import numpy as np
import networkx as nx
import math
import os
from our_agent import QAgent
import Packet
import random
import UpdateEdges as UE
from neural_network import NeuralNetwork
import matplotlib
import get_graph
import pickle

# 读取配置
main_dir = os.path.dirname(os.path.realpath(__file__))
main_dir = main_dir + '/'
with open(main_dir + 'Setting.json') as f:
    setting = json.load(f)


class dynetworkEnv(gym.Env):
    def __init__(self):
        # ---- 网络参数 ----
        self.nnodes = setting['NETWORK']['number nodes']
        self.nedges = setting['NETWORK']['edge degree']
        self.max_queue = setting['NETWORK']['holding capacity']
        self.max_transmit = setting['NETWORK']['sending capacity']
        self.max_initializations = setting['NETWORK']['max_additional_packets']
        self.npackets = setting['NETWORK']['initial num packets']
        self.max_edge_weight = setting['NETWORK']['max_edge_weight']
        self.min_edge_removal = setting['NETWORK']['min_edge_removal']
        self.max_edge_removal = setting['NETWORK']['max_edge_removal']
        self.move_number = setting['NETWORK']['node_move_number']
        self.edge_change_type = setting['NETWORK']['edge_change_type']
        self.network_type = setting['NETWORK']['network_type']
        self.initial_dynetwork = None
        self.dynetwork = None
        self.router_type = 'dijkstra'
        self.packet = -1
        self.curr_queue = []
        self.remaining = []
        self.nodes_traversed = 0
        self.print_edge_weights = True

        # ---- DQN 输入配置（原样保留）----
        self.input_q_size = setting['DQN']['take_queue_size_as_input']
        self.input_buffer_size = setting['DQN']['take_buffer_size_as_input']
        self.input_max_neighbour_buffer_size = setting['DQN']['take_max_neighbour_buffer_size_as_input']

        # ---- SP 路由（原样）----
        self.sp_packet = -1
        self.sp_curr_queue = []
        self.sp_remaining = []
        self.sp_nodes_traversed = 0
        self.preds = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = self.init_dqns()
        self.renew_nodes = []
        self.batch_size = setting['DQN']['memory_batch_size']
        self.gamma = setting['AGENT']['gamma_for_next_q_val']
        self.network_use = setting['NETWORK']['use_which_network']

        # ========= 多维度安全（CTS）=========
        ms = setting.get("MultiSecurity", {})

        # 节点静态属性
        self.attr_score = ms.get("attr_score", {"Core": 1.0, "Border": 0.7, "Unknown": 0.5, "Malicious": 0.1})
        # 若配置里没给 Malicious，兜底低分
        if "Malicious" not in self.attr_score:
            self.attr_score["Malicious"] = 0.1
        self.node_attr = ms.get("node_attr", {})

        # ======== 随机分配节点安全属性（新增）========
        attr_rand = ms.get("attr_random", {})
        self.attr_rand_enable = bool(int(attr_rand.get("enable", 0)))
        self.attr_rand_mode   = attr_rand.get("mode", "prob")  # "prob" or "count"
        self.attr_rand_probs  = attr_rand.get("probs", None)   # {"Core":0.3,"Border":0.5,"Unknown":0.2}
        self.attr_rand_counts = attr_rand.get("counts", None)  # {"Core":10,"Border":20,"Unknown":19}
        self.attr_rand_seed   = attr_rand.get("seed", None)
        self.attr_rand_resample_each_episode = bool(int(attr_rand.get("resample_each_episode", 0)))

        # 允许参与随机的类别（默认为 attr_score 去掉 "Malicious"）
        allow_default = [k for k in self.attr_score.keys() if k != "Malicious"]
        self.attr_rand_allow = attr_rand.get("allow", allow_default)

        # 明确指定的固定节点（这些节点沿用 node_attr，不参与随机）
        fixed_nodes_cfg = attr_rand.get("fixed_nodes", [])
        self.attr_rand_fixed_nodes = set(int(str(n)) for n in fixed_nodes_cfg)

        self._attr_rng = np.random.default_rng(self.attr_rand_seed)

        # Beta 信誉
        self.beta_forget = float(ms.get("beta_forget", 0.01))
        a0 = float(ms.get("beta_a0", 1.0))
        b0 = float(ms.get("beta_b0", 1.0))
        self.beta_params = {i: [a0, b0] for i in range(self.nnodes)}

        # 组合权重
        self.omega_nat = float(ms.get("omega_nat", 0.30))
        self.omega_geo = float(ms.get("omega_geo", 0.20))
        self.omega_net = float(ms.get("omega_net", 0.30))
        self.omega_rep = float(ms.get("omega_rep", 0.20))
        s = self.omega_nat + self.omega_geo + self.omega_net + self.omega_rep
        if s <= 0:
            self.omega_nat = self.omega_geo = self.omega_net = self.omega_rep = 0.25
        else:
            self.omega_nat /= s; self.omega_geo /= s; self.omega_net /= s; self.omega_rep /= s

        # 动态惩罚强度 λ_t
        self.lambda_t   = float(ms.get("lambda0", 0.2))
        self.risk_beta  = float(ms.get("risk_beta", 2.0))
        self.risk_delta = float(ms.get("risk_delta", 0.05))
        self.penalty_alpha = float(ms.get("penalty_alpha", 2.0))


        # ======== 随机恶意节点（新增）========
        mal_cfg = ms.get("malicious", {})
        self.mal_mode   = mal_cfg.get("mode", "ratio")              # "ratio" or "count"
        self.mal_ratio  = float(mal_cfg.get("ratio", 0.0))          # 比例（0~1）
        self.mal_count  = (None if mal_cfg.get("count") is None else int(mal_cfg.get("count")))
        self.mal_seed   = mal_cfg.get("seed", None)
        self.mal_resample_each_episode = bool(int(mal_cfg.get("resample_each_episode", 0)))
        self.mal_delay_factor = float(mal_cfg.get("delay_factor", 1.0))   # 恶意节点延迟放大系数（≥1.0）
        self.mal_drop_prob    = float(mal_cfg.get("drop_prob", 0.0))      # 黑洞丢包概率（0~1）
        self._mal_rng = np.random.default_rng(self.mal_seed)               # 可复现
        self._malicious_nodes = set()
        self._baseline_node_attr = dict(self.node_attr)  # 便于每轮重置

        # ---- 地图初始化（原样）----
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'q-learning/')
        if self.network_use == "new":
            print("创建新地图graph3.gpickle")
            network, positions = get_graph.new_graph2(self.nnodes)
            sp_receiving_queue_dict, sp_sending_queue_dict = {}, {}
            for i in range(self.nnodes):
                sp_receiving_queue_dict.update({i: {'sp_receiving_queue': []}})
                sp_sending_queue_dict.update({i: {'sp_sending_queue': []}})
            nx.set_node_attributes(network, sp_receiving_queue_dict)
            nx.set_node_attributes(network, sp_sending_queue_dict)
            nx.set_node_attributes(network, 0, 'sp_max_queue_len')
            nx.set_node_attributes(network, 0, 'sp_avg_q_len_array')
            nx.set_node_attributes(network, 0, 'importance')

            receiving_queue_dict, sending_queue_dict = {}, {}
            for i in range(self.nnodes):
                receiving_queue_dict.update({i: {'receiving_queue': []}})
                sending_queue_dict.update({i: {'sending_queue': []}})
            nx.set_node_attributes(network, self.max_transmit, 'max_send_capacity')
            nx.set_node_attributes(network, self.max_queue, 'max_receive_capacity')
            nx.set_node_attributes(network, self.max_queue, 'congestion_measure')
            nx.set_node_attributes(network, receiving_queue_dict)
            nx.set_node_attributes(network, sending_queue_dict)

            location_dict = {i: {'position': []} for i in range(self.nnodes)}
            nx.set_node_attributes(network, location_dict)
            for nodeIdx in network.nodes:
                node = network.nodes[nodeIdx]
                for val in positions[nodeIdx]:
                    node['position'].append(val)

            nx.set_node_attributes(network, 0, 'max_queue_len')
            nx.set_node_attributes(network, 0, 'avg_q_len_array')
            nx.set_node_attributes(network, 0, 'growth')

            for s_edge, e_edge in network.edges:
                network[s_edge][e_edge]['initial_weight'] = network[s_edge][e_edge]['edge_delay']
                network[s_edge][e_edge]['new'] = 0
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
            with open(results_dir + "graph3.gpickle", "wb") as f:
                pickle.dump(network, f)

            self.initial_dynetwork = dynetwork.DynamicNetwork(copy.deepcopy(network), self.max_initializations)
            self.dynetwork = copy.deepcopy(self.initial_dynetwork)
        else:
            print("读取已有的地图")
            with open(results_dir + "graph3.gpickle", "rb") as f:
                network = pickle.load(f)
            self.initial_dynetwork = dynetwork.DynamicNetwork(copy.deepcopy(network), self.max_initializations)
            self.dynetwork = copy.deepcopy(self.initial_dynetwork)
            positions = {nodeIdx: self.initial_dynetwork._network.nodes[nodeIdx]['position'] for nodeIdx in network.nodes}

        self.dynetwork.randomGeneratePackets(copy.deepcopy(self.npackets), False)
        self._positions = positions

        # === CTS 日志缓存 ===
        self._cts_history = []    # [(episode, t, lambda_t, avg_cts, min_cts, max_cts, avg_rep), ...]
        self._cur_episode = 0     # 当前训练轮次（由外部设置）

        # === 随机分配节点安全属性（如启用） ===
        if self.attr_rand_enable:
            self._randomize_node_attrs(initial=True)
        else:
            self._baseline_node_attr = dict(self.node_attr)

        # === 初始化并标注本轮恶意节点 ===
        self._assign_malicious_nodes(initial=True)

    # ---------- 多维度安全：组件得分 ----------
    def _beta_expected(self, node: int) -> float:
        a, b = self.beta_params[node]
        s = a + b
        if s <= 0:
            return 0.5
        return float(max(0.0, min(1.0, a / s)))

    def _beta_update(self, node: int, success: int):
        a, b = self.beta_params[node]
        a = (1.0 - self.beta_forget) * a + (1 if success else 0)
        b = (1.0 - self.beta_forget) * b + (0 if success else 1)
        eps = 1e-6
        self.beta_params[node] = [max(eps, a), max(eps, b)]

    def _nat_score(self, node: int) -> float:
        key = str(node)
        attr = self.node_attr.get(key, self.node_attr.get(node, "Unknown"))
        return float(self.attr_score.get(attr, self.attr_score.get("Unknown", 0.5)))

    def _geo_score(self, node: int) -> float:
        """
        基于“邻接边平均时延”的反向安全度：
        avg_delay_norm = mean(edge_delay) / max_edge_weight
        geo = clip(1 - avg_delay_norm, 0, 1)
        """
        nbs = list(self.dynetwork._network.neighbors(node))
        if not nbs:
            return 1.0
        delays = [float(self.dynetwork._network[node][j].get('edge_delay', self.max_edge_weight)) for j in nbs]
        avg_delay = float(np.mean(delays)) if delays else self.max_edge_weight
        denom = max(1.0, float(self.max_edge_weight))
        score = 1.0 - (avg_delay / denom)
        return float(max(0.0, min(1.0, score)))

    def _net_score(self, node: int) -> float:
        sending_queue = 'sending_queue'
        receiving_queue = 'receiving_queue'
        q = len(self.dynetwork._network.nodes[node][sending_queue]) + len(self.dynetwork._network.nodes[node][receiving_queue])
        cap = self.dynetwork._network.nodes[node]['max_receive_capacity']
        if cap <= 0:
            return 0.5
        util = min(1.0, q / cap)
        return float(max(0.0, 1.0 - util))

    def _node_cts(self, node: int) -> float:
        nat = self._nat_score(node)
        geo = self._geo_score(node)
        net = self._net_score(node)
        rep = self._beta_expected(node)
        return float(self.omega_nat * nat + self.omega_geo * geo + self.omega_net * net + self.omega_rep * rep)

    def _neighbor_penalties(self, neighbors):
        if not neighbors:
            return []
        return [self.lambda_t * ((1.0 - self._node_cts(j)) ** self.penalty_alpha) for j in neighbors]


    def _update_lambda(self):
        if self.nnodes <= 0:
            return
        avg_rep = np.mean([self._beta_expected(i) for i in range(self.nnodes)])
        risk = (1.0 - avg_rep) ** self.risk_beta
        self.lambda_t = float(max(0.0, min(1.0, (1.0 - self.risk_delta) * self.lambda_t + self.risk_delta * risk)))

    # ---------- 恶意节点相关（新增） ----------
    def _assign_malicious_nodes(self, initial=False):
        """抽样并标记恶意节点；同步刷新 node_attr 与图属性 'is_malicious'。"""
        n = self.nnodes
        if self.mal_mode == "count" and self.mal_count is not None:
            k = int(self.mal_count)
        else:  # ratio
            k = int(np.floor(self.mal_ratio * n + 1e-9))
        k = max(0, min(n, k))

        choices = self._mal_rng.choice(n, size=k, replace=False).tolist() if k > 0 else []
        self._malicious_nodes = set(int(i) for i in choices)

        # 还原到基线属性，再覆写为 Malicious
        self.node_attr = dict(self._baseline_node_attr)
        for idx in self._malicious_nodes:
            self.node_attr[str(int(idx))] = "Malicious"

        # 同步到图属性
        attrs = {int(i): {'is_malicious': 1} for i in self._malicious_nodes}
        nx.set_node_attributes(self.dynetwork._network, {i: {'is_malicious': 0} for i in self.dynetwork._network.nodes})
        if attrs:
            nx.set_node_attributes(self.dynetwork._network, attrs)

        if initial:
            print(f"[MultiSecurity] malicious nodes: {sorted(list(self._malicious_nodes))}")

    def _is_malicious(self, node: int) -> bool:
        return int(node) in self._malicious_nodes

    # ---------- 主流程 ----------
    def router(self, agent, t, will_learn=True, SP=False):
        node_queue_lengths = [0]
        num_nodes_at_capacity = 0
        num_nonEmpty_nodes = 0
        for nodeIdx in self.dynetwork._network.nodes:
            self.nodes_traversed += 1
            if self.nodes_traversed == self.nnodes:
                agent.config['update_epsilon'] = True
                self.nodes_traversed = 0

            node = self.dynetwork._network.nodes[nodeIdx]
            self.curr_queue = node['sending_queue']
            sending_capacity = node['max_send_capacity']
            holding_capacity = node['max_receive_capacity']
            queue_size = len(self.curr_queue)

            if queue_size > self.dynetwork._max_queue_length:
                self.dynetwork._max_queue_length = queue_size

            if queue_size > 0:
                node_queue_lengths.append(queue_size)
                num_nonEmpty_nodes += 1
                if queue_size > sending_capacity:
                    num_nodes_at_capacity += 1

            self.remaining = []
            sendctr = 0
            for _ in range(queue_size):
                if sendctr == sending_capacity:
                    self.dynetwork._rejections += (1 * (len(node['sending_queue'])))
                    break
                self.packet = self.curr_queue[0]
                pkt_state = self.get_state(self.packet)

                cur_state = F.one_hot(torch.tensor([pkt_state[1]]), self.nnodes)
                if self.input_q_size:
                    cur_size = torch.tensor([len(self.curr_queue)]).unsqueeze(0)
                    cur_state = torch.cat((cur_state, cur_size), dim=1)
                if self.input_buffer_size:
                    receiving_queue_size = len(self.dynetwork._network.nodes[pkt_state[1]]['receiving_queue'])
                    buffer_size = holding_capacity - sending_capacity - receiving_queue_size
                    cur_state = torch.cat((cur_state, torch.tensor([buffer_size]).unsqueeze(0)), dim=1)
                if self.input_max_neighbour_buffer_size:
                    nlist_all = sorted(list(self.dynetwork._network.neighbors(pkt_state[0])))
                    buffer_sizes = []
                    for j in nlist_all:
                        rqs = len(self.dynetwork._network.nodes[j]['receiving_queue'])
                        buffer_sizes.append(holding_capacity - sending_capacity - rqs)
                    if buffer_sizes:
                        cur_state = torch.cat((cur_state, torch.tensor([max(buffer_sizes)]).unsqueeze(0)), dim=1)

                nlist = sorted(list(self.dynetwork._network.neighbors(pkt_state[0])))

                if SP:
                    action = None if pkt_state[0] == pkt_state[1] else self.get_next_step(pkt_state[0], pkt_state[1], self.router_type)
                else:
                    if pkt_state[0] == pkt_state[1]:
                        action = None
                    else:
                        sec_penalties = self._neighbor_penalties(nlist)
                        action = agent.act(self.dqn[pkt_state[0]], cur_state, nlist, sec_penalty=sec_penalties)

                reward, self.remaining, self.curr_queue, action = self.step(action, pkt_state[0])
                if reward is not None:
                    sendctr += 1

                if will_learn and action is not None:
                    next_state = F.one_hot(torch.tensor([pkt_state[1]]), self.nnodes)
                    if self.input_q_size:
                        next_size = len(self.dynetwork._network.nodes[action]['sending_queue'])
                        next_state = torch.cat((next_state, torch.tensor([next_size]).unsqueeze(0)), dim=1).float()
                    if self.input_buffer_size:
                        rqs = len(self.dynetwork._network.nodes[action]['receiving_queue'])
                        nb = holding_capacity - sending_capacity - rqs
                        next_state = torch.cat((next_state, torch.tensor([nb]).unsqueeze(0)), dim=1).float()
                    agent.learn(self.dqn[pkt_state[0]], self.dqn, cur_state, action, reward, next_state)

            node['sending_queue'] = self.remaining + node['sending_queue']

        if len(node_queue_lengths) > 1:
            self.dynetwork._avg_q_len_arr.append(np.average(node_queue_lengths[1:]))
        self.dynetwork._num_capacity_node.append(num_nodes_at_capacity)
        self.dynetwork._num_working_node.append(num_nonEmpty_nodes)
        self.dynetwork._num_empty_node.append(self.dynetwork.num_nodes - num_nonEmpty_nodes)
        self.dynetwork._congestions.append(self.dynetwork._num_congestions)
        self.dynetwork._retransmission.append(self.dynetwork._num_retransmission)

    def updateWhole(self, agent, t, learn=True,  SP=False, savesteps=False):
        self.purgatory(False)
        self.update_queues(False)
        self.update_time(False)
        self.router(agent, t, learn, SP)
        self._update_lambda()
        # —— 记录本时间步的 CTS 快照到缓存 —— 
        avg_cts, min_cts, max_cts, avg_rep = self._snapshot_cts()
        policy = "SP" if SP else "DQN"           # <<< 新增：策略标签
        self._cts_history.append((
            int(self._cur_episode),              # episode
            int(t),                              # timestep
            float(self.lambda_t),                # 动态惩罚强度
            float(avg_cts),                      # 平均 CTS
            float(min_cts),                      # 最小 CTS
            float(max_cts),                      # 最大 CTS
            float(avg_rep),                      # 平均信誉
            policy                               # <<< 新增字段
        ))

    def change_network(self):
        self.dynetwork = copy.deepcopy(self.initial_dynetwork)
        renew_nodes = UE.Add1(self.dynetwork, self.move_number)
        if self.edge_change_type == 'sinusoidal':
            UE.Sinusoidal(self.dynetwork)
        elif self.edge_change_type == 'none':
            pass
        else:
            UE.Random_Walk(self.dynetwork)
        self.changed_dynetwork = dynetwork.DynamicNetwork(copy.deepcopy(self.dynetwork._network), self.max_initializations)
        self.renew_nodes = renew_nodes
        print("renew_nodes:", self.renew_nodes)

    def reset(self, curLoad=None, Change=False, SP=False):
        self.dynetwork = copy.deepcopy(self.changed_dynetwork) if Change else copy.deepcopy(self.initial_dynetwork)
        if curLoad is not None:
            self.npackets = curLoad
        self.dynetwork.randomGeneratePackets(self.npackets, SP)
        print('Environment reset')

    def purgatory(self, SP=False):
        if SP:
            temp_purgatory = copy.deepcopy(self.dynetwork.sp_purgatory)
            self.dynetwork.sp_purgatory = []
        else:
            temp_purgatory = copy.deepcopy(self.dynetwork._purgatory)
            self.dynetwork._purgatory = []
        for (index, weight) in temp_purgatory:
            self.dynetwork.GeneratePacket(index, SP, weight)

    def update_queues(self, SP=False):
        sending_queue = 'sp_sending_queue' if SP else 'sending_queue'
        receiving_queue = 'sp_receiving_queue' if SP else 'receiving_queue'
        for nodeIdx in self.dynetwork._network.nodes:
            node = self.dynetwork._network.nodes[nodeIdx]
            if not SP:
                node['growth'] = len(node[receiving_queue])
            queue = copy.deepcopy(node[receiving_queue])
            for elt in queue:
                pkt = elt[0]
                if elt[1] == 0:
                    node[sending_queue].append(pkt)
                    node[receiving_queue].remove(elt)
                else:
                    idx = node[receiving_queue].index(elt)
                    node[receiving_queue][idx] = (pkt, elt[1] - 1)

    def update_time(self, SP=False):
        sending_queue = 'sp_sending_queue' if SP else 'sending_queue'
        receiving_queue = 'sp_receiving_queue' if SP else 'receiving_queue'
        packets = self.dynetwork.sp_packets if SP else self.dynetwork._packets
        for nodeIdx in self.dynetwork._network.nodes:
            for elt in self.dynetwork._network.nodes[nodeIdx][receiving_queue]:
                pkt = elt[0]
                curr_time = packets.packetList[pkt].get_time()
                packets.packetList[pkt].set_time(curr_time + 1)
            for c_pkt in self.dynetwork._network.nodes[nodeIdx][sending_queue]:
                curr_time = packets.packetList[c_pkt].get_time()
                packets.packetList[c_pkt].set_time(curr_time + 1)

    def step(self, action, curNode=None):
        reward = None
        if action is None:
            self.curr_queue.remove(self.packet)
            self.remaining.append(self.packet)
            self.dynetwork._rejections += 1
        else:
            reward, self.curr_queue = self.send_packet(action)
        return reward, self.remaining, self.curr_queue, action

    def is_capacity(self, target_node, SP=False):
        sending_queue = 'sp_sending_queue' if SP else 'sending_queue'
        receiving_queue = 'sp_receiving_queue' if SP else 'receiving_queue'
        total_queue_len = len(self.dynetwork._network.nodes[target_node][sending_queue]) + \
                          len(self.dynetwork._network.nodes[target_node][receiving_queue])
        return total_queue_len >= self.dynetwork._network.nodes[target_node]['max_receive_capacity']

    def send_packet(self, next_step):
        pkt = self.dynetwork._packets.packetList[self.packet]
        curr_node = pkt.get_curPos()
        dest_node = pkt.get_endPos()
        weight = self.dynetwork._network[curr_node][next_step]['edge_delay']

        # —— 恶意节点附加行为（延迟放大/黑洞丢包）——
        if self._is_malicious(next_step):
            if self.mal_delay_factor != 1.0:
                weight = float(weight) * self.mal_delay_factor
            if (self.mal_drop_prob > 0.0) and (self._mal_rng.random() < self.mal_drop_prob):
                # 按“拥塞丢弃”的逻辑处理计数与再生成，便于沿用你的统计口径
                self.dynetwork._num_congestions += 1
                self.dynetwork._packets.packetList[self.packet]._flag = -1
                if self.dynetwork._initializations < self.dynetwork._max_initializations:
                    self.dynetwork.GeneratePacket(self.packet, False, 0, True)
                self.curr_queue.remove(self.packet)
                reward = -200
                self._beta_update(next_step, success=0)
                return reward, self.curr_queue

        # 多维安全惩罚：λ_t * (1 - CTS(next_step))
        pow_term = max(0.0, 1.0 - self._node_cts(next_step))
        sec_penalty_val = self.lambda_t * (pow_term ** self.penalty_alpha)

        # 拥塞/重传（失败）
        receiving_capacity = self.max_queue - self.max_transmit
        if len(self.dynetwork._network.nodes[pkt.get_curPos()]['receiving_queue']) >= receiving_capacity:
            self.dynetwork._packets.packetList[self.packet]._times += 1
            if self.dynetwork._packets.packetList[self.packet]._times < 10:
                self.curr_queue.remove(self.packet)
                self.remaining.append(self.packet)
                self.dynetwork._num_retransmission += 1
            else:
                self.dynetwork._num_congestions += 1
                self.dynetwork._packets.packetList[self.packet]._flag = -1
                if self.dynetwork._initializations < self.dynetwork._max_initializations:
                    self.dynetwork.GeneratePacket(self.packet, False, 0, True)
                self.curr_queue.remove(self.packet)
            reward = -200
            self._beta_update(next_step, success=0)
            return reward, self.curr_queue

        # 正常转发
        self.dynetwork._packets.packetList[self.packet].set_time(pkt.get_time() + weight)
        pkt.set_curPos(next_step)
        if pkt.get_curPos() == dest_node:
            # 成功送达
            self.dynetwork._delivery_times.append(self.dynetwork._packets.packetList[self.packet].get_time())
            self.dynetwork._deliveries += 1
            self.dynetwork._packets.packetList[self.packet]._flag = 1
            if self.dynetwork._initializations < self.dynetwork._max_initializations:
                self.dynetwork.GeneratePacket(self.packet, False, 0, True)
            self.curr_queue.remove(self.packet)
            reward = 1000
            reward -= sec_penalty_val
            self._beta_update(next_step, success=1)
        else:
            self.curr_queue.remove(self.packet)
            try:
                q = len(self.dynetwork._network.nodes[next_step]['sending_queue']) + \
                    len(self.dynetwork._network.nodes[next_step]['receiving_queue'])
                q_eq = 0.8 * self.max_queue
                w = 5
                growth = self.dynetwork._network.nodes[next_step]['growth']
                reward = 0.03 * (-(q - q_eq + w * growth))
                reward = weight * -1 + reward
            except nx.NetworkXNoPath:
                reward = -1000
            self.dynetwork._network.nodes[next_step]['receiving_queue'].append((self.packet, weight))
            reward -= sec_penalty_val
            self._beta_update(next_step, success=1)

        return reward, self.curr_queue

    def get_state(self, pktIdx):
        pkt = self.dynetwork._packets.packetList[self.packet]
        return (pkt.get_curPos(), pkt.get_endPos())

    def calc_avg_delivery(self, SP=False):
        if SP:
            delivery_times = self.dynetwork.sp_delivery_times
            avg = sum(delivery_times) / len(delivery_times)
        else:
            try:
                avg = sum(self.dynetwork._delivery_times) / len(self.dynetwork._delivery_times)
            except:
                avg = None
        return avg

    def init_dqns(self):
        temp_dqns = []
        for i in range(self.nnodes):
            if self.input_q_size:
                temp_dqns.append(NeuralNetwork(i, self.nnodes, self.input_q_size))
            if self.input_buffer_size:
                temp_dqns.append(NeuralNetwork(i, self.nnodes, self.input_buffer_size))
            if self.input_max_neighbour_buffer_size:
                temp_dqns.append(NeuralNetwork(i, self.nnodes, self.input_max_neighbour_buffer_size))
        return temp_dqns

    def update_target_weights(self):
        for nn in self.dqn:
            nn.target_net.load_state_dict(nn.policy_net.state_dict())

    def clean_replay_memories(self):
        for nn in self.dqn:
            nn.replay_memory.clean()

    def save(self, opt, model_path):
        if opt != 1:
            return
        states = {}
        for nn in self.dqn:
            key_m = f"model{nn.ID}_dict"
            key_o = f"optimizer{nn.ID}_dict"
            states[key_m] = nn.policy_net.state_dict()
            states[key_o] = nn.optimizer.state_dict()
        torch.save(states, model_path)
        print(f"当前模型存储的路径是 '{model_path}'")

    def load(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在：{model_path}")
        print(f"读取模型文件：{model_path}")
        self.dqn = self.init_dqns()
        checkpoint = torch.load(model_path, weights_only=False)
        for nn in self.dqn:
            model_key = f"model{nn.ID}_dict"
            opt_key = f"optimizer{nn.ID}_dict"
            nn.policy_net.load_state_dict(checkpoint[model_key])
            nn.target_net.load_state_dict(checkpoint[model_key])
            nn.optimizer.load_state_dict(checkpoint[opt_key])

    # ------- 绘图（原样）-------
    def render(self, i):
        node_labels = {node: node for node in self.dynetwork._network.nodes}
        nx.draw(self.dynetwork._network, pos=self._positions, labels=node_labels, node_size=200,
                font_size=8, font_weight='bold', edge_color='k')
        if self.print_edge_weights:
            edge_labels = nx.get_edge_attributes(self.dynetwork._network, 'edge_delay')
            nx.draw_networkx_edge_labels(self.dynetwork._network, pos=self._positions,
                                         edge_labels=edge_labels, label_pos=0.5, font_size=8)
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'network_images/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        plt.axis('off')
        plt.figtext(0.1, 0.1, "initial injections: " + str(i))
        plt.savefig("network_images/dynet" + str(7 * 7) + ".png")
        plt.clf()

    def draw(self, i, currTrial):
        node_labels = {node: node for node in self.dynetwork._network.nodes}
        positions = {nodeIdx: self.dynetwork._network.nodes[nodeIdx]['position'] for nodeIdx in self.dynetwork._network.nodes}
        nx.draw_networkx_nodes(self.dynetwork._network.nodes, pos=positions, node_size=400)
        edge_list, edge_list1 = [], []
        for s_edge, e_edge in self.dynetwork._network.edges:
            if self.dynetwork._network[s_edge][e_edge]['new'] == 1:
                edge_list.append([s_edge, e_edge])
            else:
                edge_list1.append([s_edge, e_edge])
        nx.draw_networkx_edges(self.dynetwork._network, pos=positions, edgelist=edge_list, edge_color='r')
        nx.draw_networkx_edges(self.dynetwork._network, pos=positions, edgelist=edge_list1, edge_color='k')
        nx.draw_networkx_labels(self.dynetwork._network.nodes, pos=positions, labels=node_labels, font_size=15, font_color='k')
        if self.print_edge_weights:
            edge_labels = nx.get_edge_attributes(self.dynetwork._network, 'edge_delay')
            nx.draw_networkx_edge_labels(self.dynetwork._network, pos=positions, edge_labels=edge_labels)
        script_dir_1 = os.path.dirname(__file__)
        results_dir_1 = os.path.join(script_dir_1, 'images/')
        if not os.path.isdir(results_dir_1):
            os.makedirs(results_dir_1)
        plt.axis('off')
        plt.figtext(0.1, 0.1, "curLoad: " + str(i))
        plt.savefig("images/dynet" + str(i) + "_" + str(currTrial) + ".png")
        plt.clf()

    def get_next_step(self, currPos, destPos, router_type):
        if len(nx.dijkstra_path(self.dynetwork._network, currPos, destPos, weight='edge_delay')) == 1:
            print("没有下一跳")
            return None
        else:
            return nx.dijkstra_path(self.dynetwork._network, currPos, destPos, weight='edge_delay')[1]

    def router_test(self, agent, will_learn=True):
        node_queue_lengths = [0]
        num_nodes_at_capacity = 0
        num_nonEmpty_nodes = 0
        for nodeIdx in self.dynetwork._network.nodes:
            self.nodes_traversed += 1
            if self.nodes_traversed == len(self.dynetwork._network.nodes):
                agent.config['update_epsilon'] = True
                self.nodes_traversed = 0
            node = self.dynetwork._network.nodes[nodeIdx]
            self.curr_queue = node['sending_queue']
            sending_capacity = node['max_send_capacity']
            holding_capacity = node['max_receive_capacity']
            queue_size = len(self.curr_queue)
            if queue_size > self.dynetwork._max_queue_length:
                self.dynetwork._max_queue_length = queue_size
            if queue_size > 0:
                node_queue_lengths.append(queue_size)
                num_nonEmpty_nodes += 1
                if queue_size > sending_capacity:
                    num_nodes_at_capacity += 1
            self.remaining = []
            sendctr = 0
            for _ in range(queue_size):
                if sendctr == sending_capacity:
                    self.dynetwork._rejections += (1 * (len(node['sending_queue'])))
                    break
                self.packet = self.curr_queue[0]
                pkt_state = self.get_state(self.packet)
                nlist = sorted(list(self.dynetwork._network.neighbors(pkt_state[0])))
                cur_state = F.one_hot(torch.tensor([pkt_state[1]]), self.nnodes)
                if self.input_q_size:
                    cur_size = torch.tensor([len(self.curr_queue)]).unsqueeze(0)
                    cur_state = torch.cat((cur_state, cur_size), dim=1)

                sec_penalties = self._neighbor_penalties(nlist)
                action = agent.act(self.dqn[pkt_state[0]], cur_state, nlist, sec_penalty=sec_penalties)

                reward, self.remaining, self.curr_queue, action = self.step(action, pkt_state[0])
                if reward is not None:
                    sendctr += 1
                if will_learn and action is not None:
                    next_state = F.one_hot(torch.tensor([pkt_state[1]]), self.nnodes)
                    if self.input_q_size:
                        next_size = len(self.dynetwork._network.nodes[action]['sending_queue'])
                        next_state_tensor = torch.tensor([next_size]).unsqueeze(0)
                        next_state = torch.cat((next_state, next_state_tensor), dim=1).float()
                    for idx in self.renew_nodes:
                        if idx == pkt_state[0]:
                            agent.learn(self.dqn[pkt_state[0]], self.dqn, cur_state, action, reward, next_state)

            node['sending_queue'] = self.remaining + node['sending_queue']
        if len(node_queue_lengths) > 1:
            self.dynetwork._avg_q_len_arr.append(np.average(node_queue_lengths[1:]))
        self.dynetwork._num_capacity_node.append(num_nodes_at_capacity)
        self.dynetwork._num_working_node.append(num_nonEmpty_nodes)
        self.dynetwork._num_empty_node.append(self.dynetwork.num_nodes - num_nonEmpty_nodes)
        self.dynetwork._congestions.append(self.dynetwork._num_congestions)

    # -------- CTS 日志工具 --------
    def begin_episode(self, ep: int):
        """在每个 episode 开始前由外部调用：标记轮次，并按需重采'安全属性'与'恶意节点'。"""
        self._cur_episode = int(ep)
        # 先按需重采“安全属性”
        if self.attr_rand_enable and self.attr_rand_resample_each_episode:
            self._randomize_node_attrs(initial=False)

        # 再按需重采“恶意节点”；若恶意集合不重采但属性变了，需重新覆盖
        if self.mal_resample_each_episode:
            self._assign_malicious_nodes(initial=False)
            print(f"[Episode {ep}] malicious nodes: {sorted(list(self._malicious_nodes))}")
        elif self.attr_rand_enable and self.attr_rand_resample_each_episode:
            self._apply_malicious_overlay()


    def _snapshot_cts(self):
        """采样当前全网 CTS 统计和平均信誉。"""
        cts_vals = [self._node_cts(i) for i in range(self.nnodes)]
        avg_cts = float(np.mean(cts_vals)) if cts_vals else 0.0
        min_cts = float(np.min(cts_vals)) if cts_vals else 0.0
        max_cts = float(np.max(cts_vals)) if cts_vals else 0.0
        avg_rep = float(np.mean([self._beta_expected(i) for i in range(self.nnodes)])) if self.nnodes > 0 else 0.5
        return avg_cts, min_cts, max_cts, avg_rep

    def pop_cts_history(self):
        """取出并清空本轮缓存，给上层写入文件。"""
        hist = self._cts_history
        self._cts_history = []
        return hist
    
    def _randomize_node_attrs(self, initial=False):
        """
        按配置为“非固定节点”随机分配安全属性（不含 'Malicious'）。
        结果写入 self._baseline_node_attr 与 self.node_attr。
        """
        # 以 Setting.json 里的 node_attr 作为固定基线
        base = {str(k): v for k, v in self.node_attr.items()}
        fixed_nodes = set(int(k) for k in base.keys()) | self.attr_rand_fixed_nodes
        candidates = [i for i in range(self.nnodes) if i not in fixed_nodes]

        if not candidates or not self.attr_rand_enable:
            self._baseline_node_attr = dict(base)
            self.node_attr = dict(self._baseline_node_attr)
            return

        # 参与随机的类别（排除 Malicious）
        classes = [c for c in self.attr_rand_allow if c in self.attr_score and c != "Malicious"]
        if not classes:
            classes = [c for c in self.attr_score.keys() if c != "Malicious"]

        if self.attr_rand_mode == "count" and isinstance(self.attr_rand_counts, dict) and self.attr_rand_counts:
            # 定额模式：构造“多重集”后无放回采样
            bag = []
            for c in classes:
                bag += [c] * int(max(0, self.attr_rand_counts.get(c, 0)))
            if len(bag) < len(candidates):
                bag += [classes[-1]] * (len(candidates) - len(bag))
            picks = self._attr_rng.choice(bag, size=len(candidates), replace=False).tolist()
        else:
            # 概率模式（默认等概率）
            probs = [float(self.attr_rand_probs.get(c, 0.0)) if self.attr_rand_probs else (1.0/len(classes)) for c in classes]
            s = sum(probs)
            probs = [p/s for p in probs] if s > 0 else [1.0/len(classes)] * len(classes)
            picks = self._attr_rng.choice(classes, size=len(candidates), replace=True, p=probs).tolist()

        for node, label in zip(candidates, picks):
            base[str(node)] = label

        self._baseline_node_attr = dict(base)
        self.node_attr = dict(self._baseline_node_attr)
        if initial:
            from collections import Counter
            cnt = Counter(self.node_attr.values())
            print("[MultiSecurity] randomized node_attr:", dict(cnt))

    def _apply_malicious_overlay(self):
        """
        在 baseline 上覆盖恶意节点为 'Malicious'（不改变恶意集合）。
        """
        self.node_attr = dict(self._baseline_node_attr)
        for idx in self._malicious_nodes:
            self.node_attr[str(int(idx))] = "Malicious"
