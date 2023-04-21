import csv
from pathlib import Path

import arguebuf

input = Path("../arguegen/data/input")
output = Path("./data/requests")
prefix = "microtexts"

for folder in input.glob("*/*/*"):
    if folder.is_dir():
        expert = folder.parent.parent.name
        topic = folder.parent.name
        case_name = folder.name

        query = arguebuf.load.file(folder / "query.txt")
        adaptations = []
        ranking = {
            f"{prefix}/{case.name}": 1 if case.name == case_name else 2
            for case in folder.parent.iterdir()
            if case.is_dir()
        }

        with Path(folder / "rules.csv").open("r") as fp:
            reader = csv.reader(fp, delimiter=",")

            for row in reader:
                adaptations.append({"source": row[0], "target": row[1]})

        query.userdata = {
            "cbrEvaluations": [
                {
                    "name": expert,
                    "generalizations": {f"{prefix}/{case_name}": adaptations},
                    "ranking": ranking,
                }
            ]
        }
        case_output = output / f"{prefix}-generalization" / expert / topic
        case_output.mkdir(parents=True, exist_ok=True)

        arguebuf.dump.file(
            query,
            case_output / f"{case_name}.json",
        )
