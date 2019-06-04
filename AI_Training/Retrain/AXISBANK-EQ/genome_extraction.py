
import neat
from neat.six_util import iteritems, itervalues
import pickle

selected_genomes = [31, 829, 785, 594, 937, 929, 942, 99, 2, 11, 17, 20, 32,
                    48, 68, 115, 134, 150, 158, 183, 209, 222, 303, 358, 387, 396]

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