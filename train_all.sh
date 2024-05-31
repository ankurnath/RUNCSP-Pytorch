

#!/bin/bash

# Run training for various graph distributions

python train_maxcut.py --distribution rnd_graph_800vertices_unweighted
python train_maxcut.py --distribution rnd_graph_800vertices_weighted
# python train_maxcut.py --distribution ER_200
# python train_maxcut.py --distribution BA_200
# python train_maxcut.py --distribution dense_MC_100_200vertices_unweighted
# python train_maxcut.py --distribution HomleKim_200vertices_unweighted
# python train_maxcut.py --distribution HomleKim_200vertices_weighted
# python train_maxcut.py --distribution Physics
# python train_maxcut.py --distribution planar_800vertices_unweighted
# python train_maxcut.py --distribution planar_800vertices_weighted
# python train_maxcut.py --distribution toroidal_grid_2D_800vertices_weighted
# python train_maxcut.py --distribution SK_spin_70_100vertices_weighted
# python train_maxcut.py --distribution WattsStrogatz_200vertices_unweighted
# python train_maxcut.py --distribution WattsStrogatz_200vertices_weighted




