
import neat
from neat.six_util import iteritems, itervalues
import pickle


selected_genomes = [941, 929, 871, 942, 99, 785, 594, 245, 1042, 927, 829,
                    7, 12, 37, 36, 53, 81, 93, 120, 136, 153, 165, 176, 181,
                    196, 250, 275, 291, 294, 316, 339, 373, 385, 404]

print("Total len: ",len(selected_genomes))

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
