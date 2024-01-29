import glob
import json
import os


def _transform(distribution: dict) -> dict:
    if distribution["distribution_name"] == "fix":
        return {
            "distribution_name": "fix",
            "distribution_params": [
                {"value": distribution["distribution_params"][0]["value"]}
            ],
        }
    elif distribution["distribution_name"] == "norm":
        return distribution  # No changes, normal follows same format
    elif distribution["distribution_name"] == "expon":
        return {
            "distribution_name": "expon",
            "distribution_params": [
                {"value": distribution["distribution_params"][1]["value"] + distribution["distribution_params"][0]["value"]},
                {"value": distribution["distribution_params"][2]["value"]},
                {"value": distribution["distribution_params"][3]["value"]},
            ],
        }
    else:
        print("Unexpected distribution!!! Dunno how to parse it!!")
        return {}


def _transform_old_distrib_to_new_distrib():
    path = "../inputs/synthetic/"
    for filename in glob.glob(os.path.join(path, '*.json')):  # only process .JSON files in folder.
        # Print info
        print(f"Processing {filename}")
        # Read parameters
        with open(filename, encoding='utf-8', mode='rt') as currentFile:
            data = currentFile.read().replace('\n', '')
            parameters = json.loads(data)
        # Transform parameters
        parameters["arrival_time_distribution"] = _transform(parameters["arrival_time_distribution"])
        for element in parameters["task_resource_distribution"]:
            for resource in element["resources"]:
                new_distrib = _transform(resource)
                resource["distribution_name"] = new_distrib["distribution_name"]
                resource["distribution_params"] = new_distrib["distribution_params"]
        for event in parameters["event_distribution"]:
            new_distrib = _transform(event)
            event["distribution_name"] = new_distrib["distribution_name"]
            event["distribution_params"] = new_distrib["distribution_params"]
        # Write transformed parameters
        with open(filename, encoding='utf-8', mode='wt') as currentFile:
            json.dump(parameters, currentFile)


if __name__ == '__main__':
    _transform_old_distrib_to_new_distrib()
