# draw_plots.py
import matplotlib.pyplot as plt
import json
import os
import numpy as np

with open('Setting.json') as f:
    setting = json.load(f)

numEpisode = setting["Simulation"]["training_episodes"]
starting_size = setting["Simulation"]["test_network_load_min"]
ending_size = setting["Simulation"]["test_network_load_max"] + \
    setting["Simulation"]["test_network_load_step_size"]
step_size = setting["Simulation"]["test_network_load_step_size"]
network_load = np.arange(starting_size,ending_size, step_size)
trials = setting["Simulation"]["test_trials_per_load"]

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'plots-hybrid/')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)
learn_results_dir = os.path.join(script_dir, 'plots-hybrid/learnRes/')
if not os.path.isdir(learn_results_dir):
    os.makedirs(learn_results_dir)
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

def learning_plot_avg_deliv(avg_deliv_learning):
    print("Average Delivery Time during learing")
    print(avg_deliv_learning)
    plt.clf()
    plt.title("Average Delivery Time Per Episode")
    plt.scatter(list(range(1, numEpisode + 1)), avg_deliv_learning)
    plt.xlabel('Episode')
    plt.ylabel('Avg Delivery Time (in steps)')
    plt.savefig(learn_results_dir + "avg_deliv_learning.png")
    plt.clf()

def learning_plot_deliv_ratio(deliv_ratio_learning):
    print("delivery_ratio during learning")
    print(deliv_ratio_learning)
    plt.clf()
    plt.title(" Delivery ratio Per Episode")
    plt.scatter(list(range(1, numEpisode + 1)), deliv_ratio_learning)
    plt.xlabel('Episode')
    plt.ylabel('Delivery ratio')
    plt.savefig(learn_results_dir + "deliv_ratio_learning.png")
    plt.clf()

def learning_plot_congestions(congestions_number_learning):
    print("congestion measure during learning")
    print(congestions_number_learning)
    plt.clf()
    plt.title("Congestion measure Per Episode")
    plt.scatter(list(range(1, numEpisode + 1)), congestions_number_learning)
    plt.xlabel('Episode')
    plt.ylabel('number of congestion')
    plt.savefig(learn_results_dir + "congestion_measure_learning.png")
    plt.clf()

def learning_plot_retransmission_ratios(retransmission_ratio_learning):
    print("retransmission_ratio during learning")
    print(retransmission_ratio_learning)
    plt.clf()
    plt.title("Retransmission_ratio Per Episode")
    plt.scatter(list(range(1, numEpisode + 1)), retransmission_ratio_learning)
    plt.xlabel('Episode')
    plt.ylabel('Retransmission_ratio')
    plt.savefig(learn_results_dir + "retransmission_ratio_learning.png")
    plt.clf()

def testing_plot_avg_deliv(all_dqn_avg_delivs, all_sp_avg_delivs):
    print("Average Delivery Time for different network-load")
    print(np.around(np.array(all_dqn_avg_delivs), 3))
    print("SP--Average Delivery Time for different network-load")
    print(np.around(np.array(all_sp_avg_delivs), 3))
    plt.clf()
    plt.figure(figsize=(20, 10), dpi=100)
    plt.title("Average Delivery Time vs Network Load",fontsize = 20)
    plt.plot(network_load, all_dqn_avg_delivs, c='red',label='Collaborate_DQN')
    plt.plot(network_load, all_sp_avg_delivs, c='blue',label='SP')
    plt.legend(loc="upper right",prop = {'size':12})
    plt.xlabel('Number of packets',fontsize=20)
    plt.ylabel('Avg Delivery Time (in steps)',fontsize=20)
    plt.xlim(0, 1750)
    plt.ylim(0, 300)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='-', alpha=0.5)
    plt.savefig(results_dir + "avg_deliv_time_testing.png")
    plt.clf()

def testing_plot_deliv_ratio(all_dqn_avg_deliv_ratios,all_sp_avg_deliv_ratios):
    print("Average Delivery ratio")
    print(np.around(np.array(all_dqn_avg_deliv_ratios), 3))
    print("SP--Average Delivery ratio")
    print(np.around(np.array(all_sp_avg_deliv_ratios), 3))
    plt.clf()
    plt.figure(figsize=(20, 10), dpi=100)
    plt.title("Average Delivery ratio vs Network Load",fontsize = 20)
    plt.plot(network_load, all_dqn_avg_deliv_ratios, c='red',label='Collaborate_DQN')
    plt.plot(network_load, all_sp_avg_deliv_ratios, c='blue',label='SP')
    plt.xlabel('Number of packets',fontsize = 20)
    plt.ylabel('ratio',fontsize = 20)
    plt.xlim(0, 1750)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc="upper right",prop = {'size':12})
    plt.grid(True, linestyle='-', alpha=0.5)
    plt.savefig(results_dir + "avg_deli_ratio_testing.png")
    plt.clf()

def testing_plot_retransmission_ratios(all_dqn_retransmission_ratios,all_sp_retransmission_ratios):
    print("Retransmission_ratios for different network-load")
    print(np.around(np.array(all_dqn_retransmission_ratios), 3))
    print("SP--Retransmission_ratios for different network-load")
    print(np.around(np.array(all_sp_retransmission_ratios), 3))
    plt.clf()
    plt.figure(figsize=(20, 10), dpi=100)
    plt.title("Retransmission_ratios vs Network Load",fontsize = 20)
    plt.plot(network_load, all_dqn_retransmission_ratios, c='red', label='Collaborate_DQN')
    plt.plot(network_load, all_sp_retransmission_ratios, c='blue', label='SP')
    plt.xlabel('Number of packets',fontsize=20)
    plt.ylabel('Retransmission_ratios',fontsize=20)
    plt.xlim(0, 1750)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc="upper right",prop = {'size':12})
    plt.grid(True, linestyle='-', alpha=0.5)
    plt.savefig(results_dir + "Retransmission_ratios_testing.png")
    plt.clf()

def testing_plot_congestions(all_dqn_congestions_numbers,all_sp_congestions_numbers):
    print("Packet_Loss_Number for different network-load")
    print(np.array(all_dqn_congestions_numbers).astype(int))
    print("SP--Packet_Loss_Number for different network-load")
    print(np.array(all_sp_congestions_numbers).astype(int))
    plt.clf()
    plt.figure(figsize=(20, 10), dpi=100)
    plt.title("Packet_Loss_Number vs Network Load",fontsize=20)
    plt.plot(network_load, all_dqn_congestions_numbers, c='red', label='Collaborate_DQN')
    plt.plot(network_load, all_sp_congestions_numbers, c='blue', label='SP')
    plt.xlabel('Number of packets', fontsize=20)
    plt.ylabel('Packet_Loss_Number', fontsize=20)
    plt.xlim(0, 1750)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc="upper right",prop = {'size':12})
    plt.grid(True, linestyle='-', alpha=0.5)
    plt.savefig(results_dir + "Packet_Loss_Number_testing.png")
    plt.clf()

def draw_learning(avg_deliv_learning,deliv_ratio_learning,congestions_number_learning,retransmission_ratio_learning):
    learning_plot_avg_deliv(avg_deliv_learning)
    learning_plot_deliv_ratio(deliv_ratio_learning)
    learning_plot_congestions(congestions_number_learning)
    learning_plot_retransmission_ratios(retransmission_ratio_learning)

def draw_testing(all_dqn_avg_delivs, all_sp_avg_delivs,all_dqn_avg_deliv_ratios,all_sp_avg_deliv_ratios,all_dqn_retransmission_ratios,all_sp_retransmission_ratios,all_dqn_congestions_numbers, all_sp_congestions_numbers):
    testing_plot_avg_deliv(all_dqn_avg_delivs, all_sp_avg_delivs)
    testing_plot_deliv_ratio(all_dqn_avg_deliv_ratios,all_sp_avg_deliv_ratios)
    testing_plot_retransmission_ratios(all_dqn_retransmission_ratios,all_sp_retransmission_ratios)
    testing_plot_congestions(all_dqn_congestions_numbers, all_sp_congestions_numbers)

def testing_plot_cts_series(dqn_series, sp_series):
    """
    dqn_series: (timesteps, avg_cts_list) for DQN
    sp_series:  (timesteps, avg_cts_list) for SP
    """
    t1, y1 = dqn_series
    t2, y2 = sp_series
    print("CTS(DQN) len:", len(y1), "CTS(SP) len:", len(y2))
    plt.clf()
    plt.figure(figsize=(20, 10), dpi=100)
    plt.title("Average CTS vs Time (DQN vs SP)", fontsize=20)
    plt.plot(t1, y1, label='DQN - Avg CTS')
    plt.plot(t2, y2, label='SP  - Avg CTS')
    plt.xlabel('Timestep', fontsize=20)
    plt.ylabel('Average CTS', fontsize=20)
    plt.ylim(0, 1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc="lower right", prop={'size': 12})
    plt.grid(True, linestyle='-', alpha=0.5)
    plt.savefig(results_dir + "cts_dqn_vs_sp.png")
    plt.clf()

def testing_plot_cts_vs_load(all_dqn_avg_cts, all_sp_avg_cts):
    """
    画 DQN 与 SP 在不同网络负载（Number of packets）下的平均 CTS（0~1）。
    """
    print("Average CTS vs Network Load")
    print("DQN:", np.around(np.array(all_dqn_avg_cts), 3))
    print("SP :", np.around(np.array(all_sp_avg_cts), 3))
    plt.clf()
    plt.figure(figsize=(20, 10), dpi=100)
    plt.title("Average CTS vs Number of Packets", fontsize=20)
    plt.plot(network_load, all_dqn_avg_cts, label='DQN', c='red')
    plt.plot(network_load, all_sp_avg_cts,  label='SP',  c='blue')
    plt.xlabel('Number of packets', fontsize=20)
    plt.ylabel('Average CTS', fontsize=20)
    plt.ylim(0, 1)
    plt.xlim(0, 1750)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc="lower right", prop={'size':12})
    plt.grid(True, linestyle='-', alpha=0.5)
    plt.savefig(results_dir + "cts_vs_load.png")
    plt.clf()
