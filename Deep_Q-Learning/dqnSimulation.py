# dqnSimulation.py
from math import e
import sys
from datetime import datetime
import torch
import torch.nn.functional as F
import time
from our_env3 import *
import matplotlib.pyplot as plt
import matplotlib
import json
import numpy as np
import os
import draw_plots
import pickle
import csv

with open('Setting.json') as f:
    setting = json.load(f)

now = datetime.now()
start_time = now.strftime("%H:%M:%S")
print("Current Time =", start_time)

numEpisode = setting["Simulation"]["training_episodes"]
Episode = setting["Simulation"]["testing_episodes"]
Episode1 = setting["Simulation"]["testing_episodes_local_training"]
time_steps = setting["Simulation"]["max_allowed_time_step_per_episode"]
learning_plot = setting["Simulation"]["learning_plot"]
comparison_plots = setting["Simulation"]["test_diff_network_load_plot"]
plot_opt = setting["Simulation"]["plot_routing_network"]
TARGET_UPDATE = setting["Simulation"]["num_time_step_to_update_target_network"]

starting_size = setting["Simulation"]["test_network_load_min"]
ending_size = setting["Simulation"]["test_network_load_max"] + setting["Simulation"]["test_network_load_step_size"]
step_size = setting["Simulation"]["test_network_load_step_size"]
network_load = np.arange(starting_size, ending_size, step_size)
for i in network_load:
    if i <= 0:
        print("Error: Network load must be positive.")
        sys.exit()

env = dynetworkEnv()
print(f"env.dynetwork: {env.dynetwork}")

env.reset(max(network_load))
print("max(network_load)", max(network_load))
agent = QAgent(env.dynetwork)
if agent.config['update_less'] == False:
    agent.config["update_models"][:, :] = True

if agent.config["sample_memory"] + agent.config["recent_memory"] + agent.config["priority_memory"] != 1:
    print("Error: Check memory type!")
    sys.exit()

avg_deliv_learning = []
deliv_ratio_learning =[]
congestions_number_learning = []
retransmission_ratio_learning = []

# 以“交付率”作为选择最优模型的指标
best_ratio = -1.0
best_model_path = None

f = open("experiences", "a")
model_path = "./models/best_model.pth"
train_times = setting["DQN"]["train_times"]
if train_times != 0:
    print("使用之前训练过的模型")
    print("model_path:", model_path)
    print("重新训练前，清空replay_memory")
    agent.config['epsilon'] = 0.5481219996180631
    env.clean_replay_memories()
    env.load(model_path)

# === CTS 日志文件设置 ===
CTS_LOG_PATH = "./logs/cts_log.csv"
os.makedirs(os.path.dirname(CTS_LOG_PATH), exist_ok=True)

def append_cts_rows(rows, csv_path=CTS_LOG_PATH):
    """rows: [(episode, t, lambda_t, avg_cts, min_cts, max_cts, avg_rep, policy), ...]"""
    header = ["episode", "timestep", "lambda_t", "avg_cts", "min_cts", "max_cts", "avg_rep", "policy"]
    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerows(rows)

# ===================== 训练 =====================
for i_episode in range(numEpisode):
    env.begin_episode(i_episode + 1)   # 标记当前 episode（用于 CTS 日志）
    print("---------- Episode:", i_episode+1, " ----------")
    step = []
    deliveries = []
    not_deliveries = 0
    false_generate = 0
    start = time.time()
    f.writelines(["Episode " + str(i_episode) + ":\n"])

    for t in range(time_steps):
        if (t+1) % 200 == 0:
            print("Time step", t + 1)
        env.updateWhole(agent,  t, learn=True, SP = False)
        if agent.config['update_less']:
            agent.config["update_models"][:, :] = True
            for destination_node in range(len(agent.config["update_models"][0, :])):
                agent.learn(env.dqn[destination_node], env.dqn, None, 0, 0, destination_node)
            agent.config["update_models"][:, :] = False
        step.append(t)
        deliveries.append(copy.deepcopy(env.dynetwork._deliveries))
        if (t+1) % TARGET_UPDATE == 0:
            env.update_target_weights()
        if (env.dynetwork._deliveries >= (env.npackets + env.dynetwork._max_initializations)):
            print("done! Finished in " + str(t + 1) + " time steps")
            break

    for index in env.dynetwork._packets.packetList:
        if env.dynetwork._packets.packetList[index].get_flag() == 0:
            not_deliveries += 1
    end = time.time()
    print("Epsilon", agent.config['epsilon'])
    print("pkts delivered:", env.dynetwork._deliveries)
    print("pkts not_delivered:", not_deliveries)
    print("pkts in purgatory:", len(env.dynetwork._purgatory))
    print("congestion happened,the number of dropped packets is:", env.dynetwork._congestions[-1])
    print("the number of retransmission is", env.dynetwork._retransmission[-1])
    print("the ratio of retransmission is:", env.dynetwork._retransmission[-1]/(env.dynetwork._deliveries+ env.dynetwork._retransmission[-1]))
    print("初始化的packets：", env.npackets)
    print("total packets:", env.npackets + env.dynetwork._initializations)

    delivery_ratio = env.dynetwork._deliveries / (env.dynetwork._deliveries + env.dynetwork._congestions[-1] if env.dynetwork._congestions[-1] > 0 else max(1, env.dynetwork._deliveries))
    print("delivery_ratio:", delivery_ratio)
    print("avg_delivery_time:", env.calc_avg_delivery())

    # 按“交付率”保存 checkpoint 与最优模型
    save_path = f"./models/model_ep{i_episode+1}_ratio{delivery_ratio:.3f}.pth"
    env.save(1, save_path)
    if delivery_ratio > best_ratio:
        best_ratio = delivery_ratio
        best_model_path = save_path
        env.save(1, "./models/best_model.pth")

    avg_deliv_learning.append(env.calc_avg_delivery())
    deliv_ratio_learning.append(delivery_ratio)
    congestions_number_learning.append(env.dynetwork._congestions[-1])
    retransmission_ratio_learning.append(env.dynetwork._retransmission[-1]/(env.dynetwork._deliveries+ env.dynetwork._retransmission[-1]))
    
    # —— 将本轮采样的 CTS 变化写入 CSV —— 
    cts_rows = env.pop_cts_history()
    if cts_rows:
        append_cts_rows(cts_rows)

    env.reset(max(network_load))

if learning_plot == 1:
    draw_plots.draw_learning(avg_deliv_learning, deliv_ratio_learning, congestions_number_learning, retransmission_ratio_learning)

env.save(setting["Simulation"]["whether_save"], model_path)

# ===================== 测试 =====================
test_opt = setting["Simulation"]["whether_test"]
network_opt = setting["Simulation"]["network_opt_test"]

def _rows_to_series(rows, expect_policy):
    """
    rows 既兼容8列(含policy)也兼容7列(不含policy):
    [episode, t, lambda_t, avg_cts, min_cts, max_cts, avg_rep, policy?]
    """
    filtered = []
    for r in rows:
        if len(r) >= 8:
            if r[-1] == expect_policy:
                filtered.append(r)
        elif len(r) == 7:
            filtered.append(r)  # 老格式：当作匹配
    filtered.sort(key=lambda x: x[1])  # 按 timestep 升序
    ts = [r[1] for r in filtered]
    ys = [r[3] for r in filtered]      # avg_cts
    return (ts, ys)

def _mean_cts(rows, expect_policy):
    # 兼容7列(无policy)与8列(有policy)的行格式
    vals = []
    for r in rows:
        if len(r) >= 8:
            if r[-1] == expect_policy:
                vals.append(r[3])  # r[3] == avg_cts
        elif len(r) >= 4:
            vals.append(r[3])
    return float(np.mean(vals)) if vals else 0.0

last_dqn_cts_series = None
last_sp_cts_series = None
ep_counter_for_test = numEpisode  # 测试从训练轮次之后继续编号

if test_opt == 1:
    print("进入测试部分")
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'q-learning/')
    if network_opt == 1:
        print("当前使用的network是graph1--测试SP时修改节点数目时使用")
        with open(results_dir + "graph1.gpickle", "rb") as f:
            network = pickle.load(f)
        env.initial_dynetwork = dynetwork.DynamicNetwork(copy.deepcopy(network), env.max_initializations)
        env.dynetwork = copy.deepcopy(env.initial_dynetwork)
    else:
        print("当前测试使用的network是graph3")
        with open(results_dir + "graph3.gpickle", "rb") as f:
            network = pickle.load(f)
        env.initial_dynetwork = dynetwork.DynamicNetwork(copy.deepcopy(network), env.max_initializations)
        env.dynetwork = copy.deepcopy(env.initial_dynetwork)

    now = datetime.now()
    start_time = now.strftime("%H:%M:%S")
    print("Current_Train_Time =", start_time)
    agent.config['epsilon'] = 0.01
    agent.config['decay_rate'] = 1

    # ★ 测试阶段固定“安全环境”不重采，确保 DQN 与 SP 可比
    env.attr_rand_resample_each_episode = False
    env.mal_resample_each_episode = False

    def Test(currTrial, curLoad, SP=False):
        step = []
        deliveries = []
        not_deliveries_testing = 0
        for t in range(time_steps):
            env.updateWhole(agent, t, learn=False, SP=SP)
            step.append(t)
            deliveries.append(copy.deepcopy(env.dynetwork._deliveries))
            if (env.dynetwork._deliveries >= (env.dynetwork._initializations + curLoad)):
                print("Finished trial,the delivery-ratio is 100%", currTrial)
                print("pkts delivered:", env.dynetwork._deliveries)
                print("total pkts:", env.npackets + env.dynetwork._initializations)
                break
        for index in env.dynetwork._packets.packetList:
            if env.dynetwork._packets.packetList[index].get_flag() == 0:
                not_deliveries_testing += 1
        print("pkts delivered:", env.dynetwork._deliveries)
        print("pkt not_deliveried:", not_deliveries_testing)
        print("congestion happened,the number of dropped packets is:", env.dynetwork._congestions[-1])
        print("the ratio of retransmission is:", env.dynetwork._retransmission[-1]/(env.dynetwork._deliveries + env.dynetwork._retransmission[-1]))
        print("total pkts:", curLoad + env.dynetwork._initializations)
        print("delivery ratio:",
              env.dynetwork._deliveries / (env.dynetwork._deliveries + env.dynetwork._congestions[-1]))
        avg = env.calc_avg_delivery()
        print("avg_delivery_time:", avg)
        return (env.calc_avg_delivery(),
                env.dynetwork._deliveries / (env.dynetwork._deliveries + env.dynetwork._congestions[-1]),
                env.dynetwork._congestions[-1],
                env.dynetwork._retransmission[-1]/(env.dynetwork._deliveries + env.dynetwork._retransmission[-1]))

    trials = setting["Simulation"]["test_trials_per_load"]
    SP_test_opt = setting["Simulation"]["SP_test_opt"]
    DQN_test_opt = setting["Simulation"]["DQN_test_opt"]
    SP_test_opt_change = setting["Simulation"]["SP_test_opt_change"]
    DQN_test_opt_change = setting["Simulation"]["DQN_test_opt_change"]

    all_dqn_avg_delivs, all_sp_avg_delivs = [], []
    dqn_avg_delivs, sp_avg_delivs = [], []
    all_dqn_avg_deliv_ratios, all_sp_avg_deliv_ratios = [], []
    dqn_avg_deliv_ratios, sp_avg_deliv_ratios = [], []
    all_dqn_retransmission_ratios, all_sp_retransmission_ratios = [], []
    dqn_retransmission_ratios, sp_retransmission_ratios = [], []
    all_dqn_congestions_numbers, all_sp_congestions_numbers = [], []
    dqn_congestions_numbers, sp_congestions_numbers = [], []
    # --- 新增：按网络负载聚合的平均 CTS ---
    all_dqn_avg_cts, all_sp_avg_cts = [], []
    dqn_avg_cts, sp_avg_cts = [], []

    for i in range(len(network_load)):
        curLoad = network_load[i]
        dqn_avg_delivs.append([]); sp_avg_delivs.append([])
        all_dqn_avg_delivs.append([]); all_sp_avg_delivs.append([])
        dqn_avg_deliv_ratios.append([]); sp_avg_deliv_ratios.append([])
        all_dqn_avg_deliv_ratios.append([]); all_sp_avg_deliv_ratios.append([])
        dqn_retransmission_ratios.append([]); sp_retransmission_ratios.append([])
        all_dqn_retransmission_ratios.append([]); all_sp_retransmission_ratios.append([])
        dqn_congestions_numbers.append([]); sp_congestions_numbers.append([])
        all_dqn_congestions_numbers.append([]); all_sp_congestions_numbers.append([])
        dqn_avg_cts.append([]); sp_avg_cts.append([])
        all_dqn_avg_cts.append([]); all_sp_avg_cts.append([])
        print("---------- Testing:", curLoad, " ----------")
        for currTrial in range(trials):
            print("-----currTrial:", currTrial + 1, "-----")
            env.render(curLoad)
            if DQN_test_opt == 1:
                env.reset(curLoad, False, False)
                env.load(model_path)
                print("测试节点不变的dqn的结果")

                ep_counter_for_test += 1
                env.begin_episode(ep_counter_for_test)

                dqn_avg_deliv, dqn_avg_deliv_ratio, dqn_congestions_number, dqn_retransmission_ratio = Test(currTrial, curLoad, SP=False)

                # --- 写入 DQN 的 CTS 记录 ---
                rows = env.pop_cts_history()
                if rows:
                    append_cts_rows(rows)
                    last_dqn_cts_series = _rows_to_series(rows, "DQN")
                    dqn_avg_cts[i].append(_mean_cts(rows, "DQN"))

                dqn_avg_delivs[i].append(dqn_avg_deliv)
                dqn_avg_deliv_ratios[i].append(dqn_avg_deliv_ratio)
                dqn_retransmission_ratios[i].append(dqn_retransmission_ratio)
                dqn_congestions_numbers[i].append(dqn_congestions_number)

            if SP_test_opt == 1:
                env.reset(curLoad, False, False)
                print("测试节点不变的sp的结果")

                ep_counter_for_test += 1
                env.begin_episode(ep_counter_for_test)

                sp_avg_deliv, sp_avg_deliv_ratio, sp_congestions_number, sp_retransmission_ratio = Test(currTrial, curLoad, SP=True)

                # --- 写入 SP 的 CTS 记录 ---
                rows = env.pop_cts_history()
                if rows:
                    append_cts_rows(rows)
                    last_sp_cts_series = _rows_to_series(rows, "SP")
                    sp_avg_cts[i].append(_mean_cts(rows, "SP"))

                sp_avg_delivs[i].append(sp_avg_deliv)
                sp_avg_deliv_ratios[i].append(sp_avg_deliv_ratio)
                sp_retransmission_ratios[i].append(sp_retransmission_ratio)
                sp_congestions_numbers[i].append(sp_congestions_number)

        dqn_avg_deliv_time = sum(dqn_avg_delivs[i]) / len(dqn_avg_delivs[i]) if dqn_avg_delivs[i] else 0.0
        sp_avg_deliv_time  = sum(sp_avg_delivs[i]) / len(sp_avg_delivs[i]) if sp_avg_delivs[i] else 0.0
        dqn_avg_delivery_ratio = sum(dqn_avg_deliv_ratios[i]) / len(dqn_avg_deliv_ratios[i]) if dqn_avg_deliv_ratios[i] else 0.0
        sp_avg_delivery_ratio  = sum(sp_avg_deliv_ratios[i]) / len(sp_avg_deliv_ratios[i]) if sp_avg_deliv_ratios[i] else 0.0
        dqn_retrans_ratio = sum(dqn_retransmission_ratios[i]) / len(dqn_retransmission_ratios[i]) if dqn_retransmission_ratios[i] else 0.0
        sp_retrans_ratio  = sum(sp_retransmission_ratios[i]) / len(sp_retransmission_ratios[i]) if sp_retransmission_ratios[i] else 0.0
        dqn_congest_number = sum(dqn_congestions_numbers[i]) / len(dqn_congestions_numbers[i]) if dqn_congestions_numbers[i] else 0.0
        sp_congest_number  = sum(sp_congestions_numbers[i]) / len(sp_congestions_numbers[i]) if sp_congestions_numbers[i] else 0.0

        all_dqn_avg_delivs[i].append(dqn_avg_deliv_time)
        all_sp_avg_delivs[i].append(sp_avg_deliv_time)
        all_dqn_avg_deliv_ratios[i].append(dqn_avg_delivery_ratio)
        all_sp_avg_deliv_ratios[i].append(sp_avg_delivery_ratio)
        all_dqn_retransmission_ratios[i].append(dqn_retrans_ratio)
        all_sp_retransmission_ratios[i].append(sp_retrans_ratio)
        all_dqn_congestions_numbers[i].append(dqn_congest_number)
        all_sp_congestions_numbers[i].append(sp_congest_number)

        dqn_avg_cts_val = sum(dqn_avg_cts[i]) / len(dqn_avg_cts[i]) if dqn_avg_cts[i] else 0.0
        sp_avg_cts_val  = sum(sp_avg_cts[i])  / len(sp_avg_cts[i])  if sp_avg_cts[i]  else 0.0

        all_dqn_avg_cts[i].append(dqn_avg_cts_val)
        all_sp_avg_cts[i].append(sp_avg_cts_val)  

    draw_plots.draw_testing(all_dqn_avg_delivs, all_sp_avg_delivs,
                            all_dqn_avg_deliv_ratios, all_sp_avg_deliv_ratios,
                            all_dqn_retransmission_ratios, all_sp_retransmission_ratios,
                            all_dqn_congestions_numbers, all_sp_congestions_numbers)
    
    # === 新增：按负载的 CTS 折线图（横轴为 Number of packets）===
    draw_plots.testing_plot_cts_vs_load(all_dqn_avg_cts, all_sp_avg_cts)

    # === 在测试总结图之后，补画 DQN vs SP 的 CTS 折线图 ===
    if (last_dqn_cts_series is not None and last_sp_cts_series is not None and
        len(last_dqn_cts_series[0]) > 0 and len(last_sp_cts_series[0]) > 0):
        draw_plots.testing_plot_cts_series(last_dqn_cts_series, last_sp_cts_series)
    else:
        print("[WARN] 没拿到完整的 CTS 时间序列，跳过 CTS 折线图绘制；"
              "请确认 our_env3.updateWhole 已将 policy 写入 CTS 日志，且测试阶段确实运行了 DQN 与 SP。")


print("start Time =", start_time)
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Whole End Time =", current_time)
print(f"训练结束！最优模型（按交付率）是：{best_model_path}（交付率={best_ratio:.3f}）")
