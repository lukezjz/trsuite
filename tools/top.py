import os
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fasta=", type=str, required=True, dest="fasta", default=None, help="path to fasta file")
    parser.add_argument("--clstr=", type=str, required=True, dest="clstr", default=None, help="path to cluster file")
    parser.add_argument("--type=", type=str, required=False, dest="type", default="total_score", help="loss type used to sort")   # !
    parser.add_argument("--top", type=int, required=False, dest="top", default=None, help="sequences with good loss")
    parser.add_argument("--folder", type=str, required=False, dest="folder", default=None, help="path to save fasta file")

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()

    with open(args.fasta) as fr:
        seqs = {}
        lines = fr.readlines()
        for i in range(len(lines) // 2):
            index = lines[i * 2][1:].split(",")[0]
            seq = lines[i * 2 + 1][:-1]
            seqs[index] = seq

    Seqs = []
    n = 0
    with open(args.clstr) as fr:
        content = fr.read()
        for split1 in content.split(">Cluster "):
            n += 1
            if split1:
                minima = 1000
                for split2 in split1.split("\n")[1:]:
                    if split2:
                        split = split2.split(",")
                        index = split[1][2:]
                        loss = float(split[-1][11:20])
                        if loss < minima:
                            minima = loss
                            idx = index
                # print(minima)
                Seqs.append((seqs[idx], minima, idx))

    if args.top:
        n = args.top

    if args.folder and not os.path.exists(args.folder):
        os.mkdir(args.folder)

    i = 0
    for seq, minima, idx in sorted(Seqs, key=lambda x: x[1])[:n]:
        i += 1
        print(i, idx, minima)
        print(seq)
        if args.folder:
            with open(os.path.join(args.folder, f"{idx}.fasta"), "w") as fw:
                fw.write(f">{idx}\n{seq}\n")


if __name__ == "__main__":
    main()

