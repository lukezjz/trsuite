usage: predict.py [-h] [-a= ALN] [-z= NPZ] [--trmodel= TRMODEL_DIRECTORY]

optional arguments:
  -h, --help            show this help message and exit
  -a= ALN, --aln= ALN   path to alignment file
  -z= NPZ, --npz= NPZ   path to npz file
  --trmodel= TRMODEL_DIRECTORY
                        path to trRosetta network weights
usage: make_cstfile.py [-h] -p= PDB [-s= STARTING_ID] [-c= CSTFILE]
                       [--range= RANGE] [--types= TYPES]



optional arguments:
  -h, --help            show this help message and exit
  -p= PDB, --pdb= PDB   path to pdb file
  -s= STARTING_ID, --starting_id= STARTING_ID
                        start index
  -c= CSTFILE, --cstfile= CSTFILE
                        path to constraint file
  --range= RANGE        range (hyphen separated)
  --types= TYPES        constraint types (theta,phi,dist,omega)
usage: make_resfile.py [-h] [-p PDB] [-f FASTA] [-l LENGTH] [-s STARTING_ID]
                       [-r RESFILE]



optional arguments:
  -h, --help            show this help message and exit
  -p PDB, --pdb= PDB    path to pdb file
  -f FASTA, --fasta= FASTA
                        path to fasta file
  -l LENGTH, --length= LENGTH
                        length of the sequence
  -s STARTING_ID, --starting_id= STARTING_ID
                        start index
  -r RESFILE, --resfile= RESFILE
                        path to resfile



usage: hallucinate.py [-h] [-l LENGTH] [-a ALN] [-r RESFILE] [-c CSTFILE]
                      [-o OUTPUT] [-z NPZ] [--trmodel= TRMODEL_DIRECTORY]
                      [--background= BACKGROUND_DIRECTORY]
                      [--aa_weight= AA_WEIGHT] [--cst_weight= CST_WEIGHT]
                      [--schedule= SCHEDULE]

optional arguments:
  -h, --help            show this help message and exit
  -l LENGTH, --length= LENGTH
                        sequence length
  -a ALN, --aln= ALN    path to starting alignment file
  -r RESFILE, --resfile= RESFILE
                        path to resfile
  -c CSTFILE, --cstfile= CSTFILE
                        path to constraint file
  -o OUTPUT, --output OUTPUT
                        path to output file
  -z NPZ, --npz NPZ     path to npz file
  --trmodel= TRMODEL_DIRECTORY
                        path to trRosetta network weights
  --background= BACKGROUND_DIRECTORY
                        path to background network weights
  --aa_weight= AA_WEIGHT
                        weight for the aa composition biasing loss term
  --cst_weight= CST_WEIGHT
                        weight for the constraints loss term
  --schedule= SCHEDULE  simulated annealing schedule:
                        'T0,n_steps,decrease_factor,decrease_range'
