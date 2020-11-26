import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_file=", type=str, required=True, dest="weights_file", default=None, help="path to bkgrd loss weights file")
    parser.add_argument("--range1=", type=str, required=True, dest="range1", default=None, help="range1 (hyphen separated)")
    parser.add_argument("--range2=", type=str, required=False, dest="range2", default=None, help="range2 (hyphen separated)")
    parser.add_argument("--weight_value=", type=float, required=False, dest="weight", default=1.0, help="bkgrd loss weights weight")
    parser.add_argument("--types=", type=str, required=False, dest="types", default=None, help="constraint types (theta,phi,dist,omega)")
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()

    range_split1 = args.range1.split("-")
    if len(range_split1) == 2:
        left1 = int(range_split1[0])
        right1 = int(range_split1[1])
    else:
        raise Exception("invalid pdb number range!")

    if args.range2:
        range_split2 = args.range2.split("-")
        if len(range_split2) == 2:
            left2 = int(range_split2[0])
            right2 = int(range_split2[1])
        else:
            raise Exception("invalid pdb number range!")
    else:
        left2 = left1
        right2 = right1

    if args.types:
        types = set(args.types.split(","))
    else:
        types = {"theta", "phi", "dist", "omega"}

    weight = float(args.weight)

    theta_lines, phi_lines, dist_lines, omega_lines = [], [], [], []
    for i1 in range(left1, right1 + 1):
        for i2 in range(left2, right2 + 1):
            if "theta" in types:
                theta_lines.append(f"theta   {i1}   {i2}   {weight}\n")
            if "phi" in types:
                phi_lines.append(f"phi   {i1}   {i2}   {weight}\n")
            if "dist" in types:
                dist_lines.append(f"dist   {i1}   {i2}   {weight}\n")
            if "omega" in types:
                omega_lines.append(f"omega   {i1}   {i2}   {weight}\n")

    weights_file = args.weights_file.rstrip(".weights") + ".weights"

    with open(weights_file, "w") as fw:
        fw.writelines(theta_lines)
        fw.writelines(phi_lines)
        fw.writelines(dist_lines)
        fw.writelines(omega_lines)


if __name__ == "__main__":
    main()
