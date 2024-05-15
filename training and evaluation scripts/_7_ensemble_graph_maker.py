""" Visualises the improvements made by an increasingly-sized ensemble of neural networks. """

import matplotlib.pyplot as plt
import numpy as np

fig = plt
fig.title('General Solver - Ensemble Results')
fig.xlabel('Number of models in ensemble')
fig.ylabel('Percentage of boards (%)')

fig.plot([1, 2, 3, 4], [30.8, 36.9, 42.3, 46.1], linestyle=':')
fig.plot([1, 2, 3, 4], [58.8, 60.1, 61.4, 62.8], marker='o')
fig.plot([1, 2, 3, 4], [30.8, 40.5, 44.4, 47.2], marker='o')
fig.plot([1, 2, 3, 4], [56.1, 39.8, 30.0, 23.6], marker='x')
fig.plot([1, 2, 3, 4], [56.1, 56.8, 57.2, 57.0], linestyle='--')
fig.plot([1, 2, 3, 4], [9.3, 9.1, 9.2, 9.1], linestyle='-.')
fig.grid()
fig.xticks(np.arange(1, 5, step=1))
fig.legend(['At least 1 correct solo solve',
            'Correctly solved by ensemble (mslm update)', 
            'Correctly solved by ensemble', 
            'No valid solo predictions', 
            'Invalid solo prediction rate',
            'Illegal solo prediction rate'], loc=8)
#fig.text(0.12, 0.92, 'test cat. acc. (total): {}'.format(round(1-test_scores[1],4)), fontsize=9, verticalalignment='top')
#fig.text(0.55, 0.92, 'test loss: {}'.format(round(test_scores[0],3)), fontsize=9, verticalalignment='top')
plt.savefig('training graphs/general/GS ensemble graph 2.png')


