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
from pytorch_neat.recurrent_net import RecurrentNet

# setup logging
# import logging
#
# formatter = logging.Formatter('%(message)s')

# def setup_logger(name, log_file, level=logging.INFO):
#     """Function setup as many loggers as you want"""
#
#     handler = logging.FileHandler(log_file)
#     handler.setFormatter(formatter)
#
#     logger = logging.getLogger(name)
#     logger.setLevel(level)
#     logger.addHandler(handler)
#
#     return logger
#
#
# logger = setup_logger('trainer_logger', 'training.log')
# logger.info('Training Logger!')

file_name = "G:\\AI Trading\\Code\\RayTrader_v3\\HistoricalData\\Min_data\\GAIL-EQ.csv"
data = trader_data.csv_to_df(file_name)
train_data, test_data = trader_data.split_data(data)
# signals = trader_data.get_signals(data)

env = trader_env.Weighted_Unrealized_BS_Env(train_data)

max_env_steps = len(env.data) - env.t - 1

resume = False
restore_file = "neat-checkpoint-5"


def eval_genome(genome, config):
    ob = env.reset()

    net = RecurrentNet.create(genome, config)

    current_max_fitness = 0
    fitness_current = 0
    counter = 0
    step = 0
    step_max = max_env_steps

    done = False

    while not done:

        # inputs = trader_data.get_inputs(signals, step)

        states = [ob]
        outputs = net.activate(states).numpy()

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

        if done or env.amt <= 0:
            done = True
            # print(genome_id, fitness_current)
            message = "Fitness :{} Max Fitness :{} Avg Daily Profit :{} %".format(fitness_current,
                                                                                  current_max_fitness,
                                                                                  round(mean(env.daily_profit_per), 3))
            print("Initial Value: ",2000)
            print("Final Value: ",env.amt)
            print("Days: ",len(env.daily_profit_per))
            print(message)
            plt.plot(env.daily_profit_per)
            plt.show()
            # logger.info(message)

        genome.fitness = fitness_current


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config.cfg')

genomeFile = '.\\genomes\\594.pkl'
genome = pickle.load(open(genomeFile, 'rb'))

# Single Threaded
for i in range(1):
    eval_genome(genome, config)


# threads = []
# if __name__ == "__main__":
#     # Multi-Processing
#     print("Multi-Processing")
#     for i in range(1):
#         threads.append(multiprocessing.Process(target=eval_genome, args=(genome, config,)))
#
#     # start all threads
#     for t in threads:
#         t.start()
#
#     # Join all threads
#     for t in threads:
#         t.join()
