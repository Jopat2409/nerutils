import argparse

def iob1_to_iob2(infile: str, outfile: str) -> None:
    data = ""
    with open(infile, "r", encoding='utf-8') as f:
        current_entity = "O"
        for line in f.readlines():
            lines = line.strip()
            if lines.startswith("-DOCSTART-") or not lines:
                data += line
                current_entity = "O"
            else:
                _, _, _, ent = lines.split(" ")
                if ent.startswith("I") and (current_entity == "O" or current_entity != ent[2:]):
                    data += line.replace(ent, f"B-{ent[2:]}")
                else:
                    data += line
                current_entity = ent[2:] if ent != "O" else "O"

    with open(outfile, "w+", encoding='utf-8') as f:
        f.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converts a CoNLL-formatted IOB1 file to IOB2')
    parser.add_argument('input', help="The input file")
    parser.add_argument('output', help="The file to output the IOB2 tags to")

    args = parser.parse_args()
    iob1_to_iob2(args.input, args.output)





