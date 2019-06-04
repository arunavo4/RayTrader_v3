
import neat
from neat.six_util import iteritems, itervalues
import pickle


selected_genomes = [871, 929, 335, 941, 942, 99, 365, 594, 785, 927, 829, 245, 1042,
                    22, 25, 54, 52, 53, 97, 108, 118, 168, 174, 177, 176, 206, 211,
                    222, 221, 224, 248, 264, 278, 273, 277, 287, 314, 319, 316, 321,
                    362, 371, 386, 387, 394, 405]

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
