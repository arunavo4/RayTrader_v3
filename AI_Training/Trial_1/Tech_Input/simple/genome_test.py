import glob
import multiprocessing
import trader_env
import trader_data
import visualize
import reporter
from statistics import mean
import numpy as np
import neat
import pickle
import matplotlib.pyplot as plt


file_name = "G:\\AI Trading\\Code\\RayTrader_v3\\HistoricalData\\Min_data\\ADANIPORTS-EQ.csv"
data = trader_data.csv_to_df(file_name)
train_data, test_data = trader_data.split_data(data)

env = trader_env.Weighted_Unrealized_BS_Env(train_data)
max_env_steps = len(env.data) - env.t - 1

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config.cfg')


def eval_genome(genome, config):
    global env, max_env_steps
    ob = env.reset()

    net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

    current_max_fitness = 0
    fitness_current = 0
    counter = 0
    step = 0
    step_max = max_env_steps

    done = False

    while not done:

        # inputs = trader_data.get_inputs(signals, step)

        nnOutput = net.activate(ob)

        ob, rew, done, _ = env.step(np.argmax(nnOutput))
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
            print("Initial Value: ",2000)
            print("Final Value: ",env.amt)
            print("Days: ",len(env.daily_profit_per))
            print(message)
            plt.title(genome.key)
            plt.plot(env.daily_profit_per)
            plt.show()
            # logger.info(message)

        genome.fitness = fitness_current


def run_tests(genome):
    global env, max_env_steps, config
    env = trader_env.Weighted_Unrealized_BS_Env(train_data)
    max_env_steps = len(env.data) - env.t - 1

    eval_genome(genome,config)

    env = trader_env.Weighted_Unrealized_BS_Env(test_data)
    max_env_steps = len(env.data) - env.t - 1

    eval_genome(genome,config)


def run_files(files_set):
    for genomeFile in files_set:
        genome = pickle.load(open(genomeFile, 'rb'))
        run_tests(genome)
        print("#"*50)


def chunks(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


# Load all the genomes
files = glob.glob(".\\genomes\\*.pkl")
n_processes = 3


threads = []
if __name__ == "__main__":
    # divide the file-list
    chunks_list = chunks(files, n_processes)

    for i in range(n_processes):
        threads.append(multiprocessing.Process(target=run_files, args=(chunks_list[i],)))

    # start all threads
    for t in threads:
        t.start()

    # Join all threads
    for t in threads:
        t.join()

#
# if __name__ == "__main__":
#     genomeFile = '.\\genomes\\594.pkl'
#     genome = pickle.load(open(genomeFile, 'rb'))
#     run_tests(genome)

