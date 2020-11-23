import argparse
import os
import time

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, dest="csv", default=None, help="path to csv file")
    parser.add_argument("--top", type=int, required=False, dest="top", default=None, help="sequences with good loss")
    parser.add_argument("--folder", type=str, required=False, dest="folder", default=f"{time.strftime('%Y%m%d%H%M%S')}", help="path to save fasta file")
    parser.add_argument("--type", type=str, required=False, dest="type", default="total_loss", help="loss")
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    data = []
    m = 0
    with open(args.csv) as fr:
        for line in fr.readlines():
            m += 1
            split = line.strip("\n").split(",")
            n = split[0].strip()
            T = split[1].strip()
            seq = split[2].strip()
            line = f">{0},{T}"
            value = ""
            for item in split[3:]:
                loss_type, loss_value = item.split("=")
                if loss_type == args.type:
                    value = float(loss_value)
                line += f",{loss_type.strip()}={loss_value.strip()}"
            line += f"\n{seq}"
            data.append((f"{n}.fasta", line, value))
            # print(line)

    if args.top:
        m = args.top

    if not os.path.exists(args.folder):
        os.mkdir(args.folder)

    for item in sorted(data, key=lambda x: x[-1])[:m]:
        print(item[1])
        with open(os.path.join(args.folder, item[0]), "w") as fw:
            fw.write(item[1])


if __name__ == '__main__':
    main()
