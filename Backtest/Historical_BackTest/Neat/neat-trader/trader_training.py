import trader_env
import trader_data
import visualize
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
signals = trader_data.get_signals(data)

initial_start_index = 40

env = trader_env.Weighted_Unrealized_BS_Env(train_data[initial_start_index+1:])

resume = True
restore_file = "neat-checkpoint-5"


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        ob = env.reset()

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        counter = 0
        step = initial_start_index
        step_max = initial_start_index + len(env.data) - 1

        done = False

        while not done:

            inputs = trader_data.get_inputs(signals,step)

            nnOutput = net.activate(inputs)

            ob, rew, done = env.step(np.argmax(nnOutput))
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

            if done or counter == 250:
                done = True
                # print(genome_id, fitness_current)
                message = "Genome_id # :{}  Fitness :{} Max Fitness :{}".format (genome_id, fitness_current, current_max_fitness)
                print(message)
                logger.info(message)

            genome.fitness = fitness_current


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')

if resume:
    p = neat.Checkpointer.restore_checkpoint(restore_file)
else:
    p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(5))

winner = p.run(eval_genomes)

visualize.draw_net(config, winner)
visualize.plot_stats(stats, ylog=False, view=True)
visualize.plot_species(stats, view=True)

print(winner)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)