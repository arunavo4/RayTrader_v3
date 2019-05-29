"""
###############   Parallel Trainer   ###############

# Simple Neat implementation in pytorch
# This is a Trader where it just looks into its past history nad the current position value and trades accordingly.
"""

import multiprocessing
import numpy as np
import neat
import pickle
import trader_env
import trader_data
import reporter
# import visualize

from pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.recurrent_net import RecurrentNet


file_name = "/home/ubuntu/AI_Training/data/YESBANK-EQ.csv"
data = trader_data.csv_to_df(file_name)
train_data, test_data = trader_data.split_data(data)
# signals = trader_data.get_signals(data)

env = trader_env.Weighted_Unrealized_BS_Env(train_data)

max_env_steps = len(env.data) - env.t - 1

resume = True
restore_file = "neat-checkpoint-0"


def make_env():
    return env


def make_net(genome, config, bs):
    return RecurrentNet.create(genome, config, bs)


def activate_net(net, states):
    outputs = net.activate(states).numpy()
    return np.argmax(outputs, axis=1)


def run(n_generations, n_processes):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config_path = "config.cfg"
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    evaluator = MultiEnvEvaluator(
        make_net, activate_net, make_env=make_env, max_env_steps=max_env_steps
    )

    if n_processes > 1:
        pool = multiprocessing.Pool(processes=n_processes)

        def eval_genomes(genomes, config):
            fitnesses = pool.starmap(
                evaluator.eval_genome, ((genome_id, genome, config) for genome_id, genome in genomes)
            )
            for (genome_id, genome), fitness in zip(genomes, fitnesses):
                genome.fitness = fitness

    else:
        def eval_genomes(genomes, config):
            for i, (genome_id, genome) in enumerate(genomes):
                try:
                    genome.fitness = evaluator.eval_genome(genome_id, genome, config)
                except Exception as e:
                    print(genome)
                    raise e

    if resume:
        pop = neat.Checkpointer.restore_checkpoint(restore_file)
    else:
        pop = neat.Population(config)

    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(reporter.LoggerReporter(True))
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(1))

    winner = pop.run(eval_genomes, n_generations)

    # visualize.draw_net(config, winner)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

    print(winner)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)


if __name__ == "__main__":
    run(n_generations=30, n_processes=10)

