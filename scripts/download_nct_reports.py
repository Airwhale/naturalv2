import argparse
import json
import logging
import os

import requests


logging.basicConfig(level=logging.INFO)

URL = "https://clinicaltrials.gov/api/v2/studies"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="nct_reports")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    args.data_path += "_test" if args.test else ""

    params = {
        "format": "json",
        "aggFilters": "studyType:int,results:with,status:com"
        if not args.test
        else "studyType:int,results:without,status:act",
        "countTotal": "true",
        "pageSize": "1000",
    }

    all_trials = []
    expected_num_pages = 1
    curr_page = 1

    while True:
        response = requests.get(
            URL,
            params=params,
            headers={"accept": "application/json"},
        )

        if response.status_code != 200:
            logging.error("Failed to download completed trials: " + response.text)
            break

        response_json = response.json()
        if "totalCount" in response_json:
            total_trials = response_json["totalCount"]
            logging.info("Expected number of trials: %d", total_trials)
            expected_num_pages = total_trials // 1000 + 1

        all_trials.extend(response_json["studies"])
        logging.info("Downloaded page %d of %d", curr_page, expected_num_pages)

        next_token = response_json.get("nextPageToken")
        if not next_token:
            break

        curr_page += 1
        params["pageToken"] = next_token

    logging.info("Successfully downloaded %d trials", len(all_trials))

    os.makedirs(args.data_path, exist_ok=True)

    for trial in all_trials:
        with open(
            os.path.join(
                args.data_path,
                f"{trial['protocolSection']['identificationModule']['nctId']}.json",
            ),
            "w",
        ) as f:
            json.dump(trial, f, indent=2)
