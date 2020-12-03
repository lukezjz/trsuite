import argparse
import Bio.PDB.PDBParser

# tested
# python hallucinate2/tools/make_resfile.py -f hallucinate2/test/A.fasta -o A.resfile -s 1
# python hallucinate2/tools/make_resfile.py -l 100 -o s3l100 -s 3

aa1_aa3 = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
           "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
           "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
           "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pdb=", type=str, required=False, dest="pdb", default=None, help="path to pdb file")
parser.add_argument("-f", "--fasta=", type=str, required=False, dest="fasta", default=None, help="path to fasta file")
parser.add_argument("-l", "--length=", type=int, required=False, dest="length", default=100, help="length of the sequence")
parser.add_argument("-s", "--starting_id=", type=int, required=False, dest="starting_id", default=1, help="start index")
parser.add_argument("-r", "--resfile=", type=str, required=False, dest="resfile", default=None, help="path to resfile")
args = parser.parse_args()

# resfile_lines: idx aa_allowed
if args.pdb is not None and args.fasta is None:
    resfile_lines = [f"{res.get_id()[1] + args.starting_id - 1}   {aa1_aa3[res.get_resname()]}\n" for res in Bio.PDB.PDBParser().get_structure("", args.pdb)[0]["A"]]
    resfile = args.pdb.split("/")[-1].rstrip(".pdb") + ".resfile"
elif args.pdb is None and args.fasta is not None:
    with open(args.fasta) as fr:
        lines = fr.readlines()
    if len(lines[0]) > 2 and lines[0][0] == ">" and len(lines[1]) > 1:
        seq = lines[1].rstrip()
        unk_aa = set(seq) - {"A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"}
        assert not unk_aa, "{} in {} cannot be recognized!".format(",".join(list(unk_aa)), args.fasta)
        resfile_lines = [f"{idx + args.starting_id}   {aa}\n" for idx, aa in enumerate(seq)]
        resfile = args.fasta.split("/")[-1].rstrip(".fasta") + ".resfile"
    else:
        raise Exception(f"File {args.fasta} cannot be recognized!")
elif args.pdb is None and args.fasta is None:
    resfile_lines = [f"{idx + args.starting_id}   ARNDCQEGHILKMFPSTWYV\n" for idx in range(args.length)]
    resfile = f"s{args.starting_id}l{args.length}.resfile"
else:
    raise Exception(f"pdb file and fasta file cannot be used simultaneously!")

if args.resfile is not None:
    resfile = args.resfile.rstrip(".resfile") + ".resfile"

with open(resfile, "w") as fw:
    fw.writelines(resfile_lines)
