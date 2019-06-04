
import neat
from neat.six_util import iteritems, itervalues
import pickle

selected_genomes = [927, 245, 594, 785, 871, 942, 941, 99, 929, 6, 8,
                    11, 12, 55, 93, 121, 167, 178, 179, 190, 207, 219,
                    232, 262, 278, 357, 359, 364, 394, 405]

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