import trader_env
import trader_data
import reporter
from statistics import mean
import numpy as np
import neat
import pickle

# setup logging
import logging

formatter = logging.Formatter('%(message)s')


def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


logger = setup_logger('trainer_logger', 'training.log')
logger.info('Training Logger!')

file_name = "G:\\AI Trading\\Code\\RayTrader_v3\\HistoricalData\\Min_data\\ADANIPORTS-EQ.csv"
data = trader_data.csv_to_df(file_name)
train_data, test_data = trader_data.split_data(data)
# signals = trader_data.get_signals(data)

env = trader_env.Weighted_Unrealized_BS_Env(train_data)
max_env_steps = len(env.data) - env.t - 1

resume = False
restore_file = "neat-checkpoint-4"


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
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

            if done or fitness_current < -10000 or env.amt <= 0.0:
                done = True
                if len(env.daily_profit_per) == 0:
                    avg_profit = 0.0
                else:
                    avg_profit = round(mean(env.daily_profit_per), 3)

                message = "Genome_id # :{}  Fitness :{} Max Fitness :{} Final Amt :{} Days : {} Avg Daily Profit :{} %".format(
                    genome_id,
                    round(fitness_current, 2),
                    round(current_max_fitness, 2), round(env.amt, 2), len(env.daily_profit_per),
                    avg_profit)
                print(message)
                logger.info(message)

            genome.fitness = fitness_current


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config.cfg')

if resume:
    p = neat.Checkpointer.restore_checkpoint(restore_file)
else:
    p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
p.add_reporter(reporter.LoggerReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(5))

winner = p.run(eval_genomes, 100)

# visualize.draw_net(config, winner)
# visualize.plot_stats(stats, ylog=False, view=True)
# visualize.plot_species(stats, view=True)

print(winner)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)