



# Run training for various graph distributions
#!/bin/bash

# Run training for various graph distributions
python evaluate.py --distribution Physics -network_steps 250
python evaluate.py --distribution SK_spin_70_100vertices_weighted --network_steps 200
python evaluate.py --distribution ER_200 --network_steps 400
python evaluate.py --distribution BA_200 --network_steps 400
python evaluate.py --distribution dense_MC_100_200vertices_unweighted --network_steps 400
python evaluate.py --distribution HomleKim_200vertices_unweighted --network_steps 400
python evaluate.py --distribution HomleKim_200vertices_weighted --network_steps 400
python evaluate.py --distribution HomleKim_800vertices_unweighted --network_steps 1600
python evaluate.py --distribution HomleKim_800vertices_weighted --network_steps 1600

python evaluate.py --distribution planar_800vertices_unweighted --network_steps 1600
python evaluate.py --distribution planar_800vertices_weighted --network_steps 1600
python evaluate.py --distribution toroidal_grid_2D_800vertices_weighted --network_steps 1600

python evaluate.py --distribution WattsStrogatz_200vertices_unweighted --network_steps 400
python evaluate.py --distribution WattsStrogatz_200vertices_weighted --network_steps 400
python evaluate.py --distribution WattsStrogatz_800vertices_unweighted --network_steps 1600
python evaluate.py --distribution WattsStrogatz_800vertices_weighted --network_steps 1600
python evaluate.py --distribution BA_800vertices_unweighted --network_steps 1600
python evaluate.py --distribution BA_800vertices_weighted --network_steps 1600



