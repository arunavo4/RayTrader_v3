
import neat
from neat.six_util import iteritems, itervalues
import pickle

selected_genomes = [245, 1042, 78, 76, 108, 156, 165, 167, 202, 150, 204, 233,
                    238, 248, 241, 293, 323, 364, 363, 360, 378, 405, 401]

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
