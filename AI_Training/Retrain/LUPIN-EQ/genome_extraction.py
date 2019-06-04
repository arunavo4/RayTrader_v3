
import neat
from neat.six_util import iteritems, itervalues
import pickle

selected_genomes = [929, 99, 942, 245, 785, 594, 31, 829, 9, 38, 40,
                    39, 52, 68, 70, 95, 122, 124, 232, 249, 333, 350, 358]

for i in range(30):
    restore_file = "neat-checkpoint-" + str(i)
    pop = neat.Checkpointer.restore_checkpoint(restore_file)

    for g in itervalues(pop.population):
        # print(str(g.key))
        if g.key in selected_genomes:
            print(str(g.key))

            save_path = './genomes/' + str(g.key) + '.pkl'
            with open(save_path, 'wb') as output:
                pickle.dump(g, output, 1)
