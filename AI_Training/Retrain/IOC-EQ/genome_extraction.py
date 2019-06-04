
import neat
from neat.six_util import iteritems, itervalues
import pickle

selected_genomes = [941, 929, 99, 1042, 785, 927, 829, 65, 170, 188,
                    199, 206, 236, 273, 319, 346, 349, 363]

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
