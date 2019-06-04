
import neat
from neat.six_util import iteritems, itervalues
import pickle

selected_genomes = [914, 245, 829, 937, 871, 1042, 594, 335, 785,
                    927, 1072, 941, 942, 99, 929, 10, 20, 17, 50,
                    57, 61, 74, 72, 101, 119, 158, 159, 174, 170,
                    158, 190, 186, 202, 215, 274, 283, 326, 339,
                    342, 341, 365]

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