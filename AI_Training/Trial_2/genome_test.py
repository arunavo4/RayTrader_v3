import glob
import multiprocessing
import trader_env
import trader_data
import genome_plots
from statistics import mean
import numpy as np
import neat
import pickle
import matplotlib.pyplot as plt
from pytorch_neat.recurrent_net import RecurrentNet


file_name = "G:\\AI Trading\\Code\\RayTrader_v3\\HistoricalData\\Min_data\\ADANIPORTS-EQ.csv"
data = trader_data.csv_to_df(file_name)
train_data, test_data = trader_data.split_data(data)

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config.cfg')


def eval_genome(genome, config, env_data):
    env = trader_env.Weighted_Unrealized_BS_SL_Env(env_data)
    max_env_steps = len(env.data) - env.t - 1

    ob = env.reset()

    net = RecurrentNet.create(genome, config)

    current_max_fitness = 0
    fitness_current = 0
    counter = 0
    step = 0
    step_max = max_env_steps
    actions = []
    for _ in range(env.t):
        actions.append(0)
    done = False

    while not done:

        # inputs = trader_data.get_inputs(signals, step)
        states = [ob]
        outputs = net.activate(states).numpy()
        actions.append(np.argmax(outputs))

        ob, rew, done, _ = env.step(np.argmax(outputs))
        # print("id",genome_id,"Step:",step,"act:",np.argmax(nnOutput),"reward:",rew)

        fitness_current += rew
        step += 1

        if fitness_current > current_max_fitness:
            current_max_fitness = fitness_current
            counter = 0
        else:
            counter += 1

        if step >= step_max:
            done = True

        if done or env.amt<=0:
            done = True
            print("Genome id#: ", genome.key)
            message = "Fitness :{} Max Fitness :{} Avg Daily Profit :{} %".format(fitness_current,
                                                                                  current_max_fitness,
                                                                                  round(mean(env.daily_profit_per), 3))
            print(message)

        genome.fitness = fitness_current

    return env, actions


def run_tests(genome):
    global train_data, test_data
    train_env, train_acts = eval_genome(genome,config,train_data)

    test_env, test_acts = eval_genome(genome,config,test_data)

    reward_filename = './/genome_plots//' + str(genome.key) + '_reward.png'
    genome_plots.plot_train_test_reward(train_env.daily_profit_per, test_env.daily_profit_per, reward_filename)

    actions_filename = './/genome_plots//' + str(genome.key) + '_actions.png'
    date_split = '2018-12-28'
    genome_plots.plot_train_test_actions(genome.key, train_env, test_env, train_acts, test_acts, date_split, actions_filename)


def run_files(files_set):
    for genomeFile in files_set:
        genome = pickle.load(open(genomeFile, 'rb'))
        run_tests(genome)


def chunks(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


# Load all the genomes
# files = glob.glob(".\\genomes\\*.pkl")
# n_processes = 3
#
#
# threads = []
# if __name__ == "__main__":
#     # divide the file-list
#     chunks_list = chunks(files, n_processes)
#
#     for i in range(n_processes):
#         threads.append(multiprocessing.Process(target=run_files, args=(chunks_list[i],)))
#
#     # start all threads
#     for t in threads:
#         t.start()
#
#     # Join all threads
#     for t in threads:
#         t.join()


# Single genome
if __name__ == "__main__":
    genomeFile = '.\\genomes\\1042.pkl'
    genome = pickle.load(open(genomeFile, 'rb'))
    run_tests(genome)

