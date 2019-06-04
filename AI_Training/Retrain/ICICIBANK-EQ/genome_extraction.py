
import neat
from neat.six_util import iteritems, itervalues
import pickle

selected_genomes = [829, 927, 941, 594, 785, 942, 99, 929, 18, 17, 44, 48,
                    101, 204, 221, 249, 264, 261, 278, 288, 306, 305, 316,
                    331, 345, 373, 386, 390, 403]

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
