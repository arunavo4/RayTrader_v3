
import neat
from neat.six_util import iteritems, itervalues
import pickle


selected_genomes = [245, 829, 942, 871, 99, 9, 11, 7, 1042, 37, 54, 56, 81, 96,
                    106, 112, 123, 166, 175, 247, 305, 334, 348, 358, 404]

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