import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv=", type=str, required=True, dest="csv", default=None, help="path to csv file")
    parser.add_argument("-f", "--fasta=", type=str, required=False, dest="fasta", default=None, help="path to fasta file")
    args = parser.parse_args()
    return args


# cd-hit -i fasta_file -o output_file -c 0.9 -n 5 -M 16000 -d 0 -T 8
def main():
    args = get_arguments()
    data = ""
    with open(args.csv) as fr:
        for line in fr.readlines():
            split = line.strip("\n").split(",")
            n = split[0].strip()
            T = split[1].strip()
            seq = split[2].strip()
            data += f">{n},{T}"
            for item in split[3:]:
                loss_type, loss_value = item.split("=")
                data += f",{loss_type.strip()}={loss_value.strip()}"
            data += f"\n{seq}\n"

    if args.fasta:
        fasta = args.fasta.rstrip(".fasta") + ".fasta"
    else:
        fasta = args.csv.rstrip(".csv") + ".fasta"

    with open(fasta, "w") as fw:
        fw.write(data)


if __name__ == "__main__":
    main()
