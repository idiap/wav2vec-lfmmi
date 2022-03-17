#!/usr/bin/env python3

from collections import OrderedDict
import json
import argparse
import sys
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("espresso.tools.asr_merge_wav_json")


def merge_json(obj, data_json_path):
    with open(data_json_path, "rb") as f:
        loaded_json = json.load(f, object_pairs_hook=OrderedDict)
    
    for utt_id, val in loaded_json.items():
        assert utt_id not in obj, "Duplicate utterance found"
        obj[utt_id] = val

    return obj


def main():
    parser = argparse.ArgumentParser(
        description = "Merge multiple training/validation json files into one."
    )
    # fmt: off
    parser.add_argument(
        "--json-files",
        nargs="+",
        required=True,
        help="path to the json files to be merged"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="path to save json output"
    )
    args = parser.parse_args()

    logger.info("Output file {}".format(args.output))
    logger.info("Total files to merge: {}".format(len(args.json_files)))

    obj = OrderedDict()
    for f in args.json_files:
        logger.info("Merging examples in {}".format(f))
        obj = merge_json(obj, f)

    with open(args.output, 'w') as f:
        json.dump(obj, f, indent=4)

    logger.info("Dumped {} examples in {}".format(len(obj), args.output))

if __name__ == "__main__":
    main()
