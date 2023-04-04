import json
import os

import task_dataset

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    configuration = json.load(open(os.path.join(dir_path, "configuration.json")))

    for task in configuration:
        config = configuration[task]
        dataset = task_dataset.TaskDataset(config)

        datapoint = dataset.generate_datapoint()
        
        print(task)
        print(datapoint[1])
