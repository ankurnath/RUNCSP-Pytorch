



# Run training for various graph distributions
#!/bin/bash

# Run training for various graph distributions
python eval_maxcut.py --distribution ER_200 --network_steps 400
python eval_maxcut.py --distribution BA_200 --network_steps 400
python eval_maxcut.py --distribution dense_MC_100_200vertices_unweighted --network_steps 400
python eval_maxcut.py --distribution HomleKim_200vertices_unweighted --network_steps 400
python eval_maxcut.py --distribution HomleKim_200vertices_weighted --network_steps 400
python eval_maxcut.py --distribution HomleKim_800vertices_unweighted --network_steps 1600
python eval_maxcut.py --distribution HomleKim_800vertices_weighted --network_steps 1600
# python eval_maxcut.py --distribution Physics 
python eval_maxcut.py --distribution planar_800vertices_unweighted --network_steps 1600
python eval_maxcut.py --distribution planar_800vertices_weighted --network_steps 1600
python eval_maxcut.py --distribution toroidal_grid_2D_800vertices_weighted --network_steps 1600
python eval_maxcut.py --distribution SK_spin_70_100vertices_weighted --network_steps 200
python eval_maxcut.py --distribution WattsStrogatz_200vertices_unweighted --network_steps 400
python eval_maxcut.py --distribution WattsStrogatz_200vertices_weighted --network_steps 400
python eval_maxcut.py --distribution WattsStrogatz_800vertices_unweighted --network_steps 1600
python eval_maxcut.py --distribution WattsStrogatz_800vertices_weighted --network_steps 1600
python eval_maxcut.py --distribution BA_800vertices_unweighted --network_steps 1600
python eval_maxcut.py --distribution BA_800vertices_weighted --network_steps 1600




