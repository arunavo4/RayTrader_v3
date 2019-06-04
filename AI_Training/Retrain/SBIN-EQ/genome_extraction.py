
import neat
from neat.six_util import iteritems, itervalues
import pickle

selected_genomes = [941, 871, 929, 942, 99, 1042, 245, 335, 594, 785,
                    927, 829, 1, 9, 12, 21, 27, 34, 54, 69, 70, 83, 94,
                    96, 111, 112, 126, 133, 134, 146, 163, 162, 182, 189,
                    206, 223, 224, 234, 264, 274, 318, 320, 330, 346, 349,
                    356, 376, 402, 403]

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
