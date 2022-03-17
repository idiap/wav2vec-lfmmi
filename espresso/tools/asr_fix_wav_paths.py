#!/usr/bin/env python3

from collections import OrderedDict
import json
import argparse
import os
import sys
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("espresso.tools.asr_fix_wav_paths")


def fix_path(obj, data_json_path):
    with open(data_json_path, "rb") as f:
        loaded_json = json.load(f, object_pairs_hook=OrderedDict)
    
    for utt_id, val in loaded_json.items():
        assert "wav" in val, "wav does not exist in {}".format(utt_id)
        wav = val["wav"]
        reldir = os.path.dirname(wav).split(' ')[-1]
        absdir = os.path.abspath(reldir)
        new_wav = wav.replace(reldir, absdir)
        obj[utt_id] = val
        obj[utt_id]['wav'] = new_wav

    return obj


def main():
    parser = argparse.ArgumentParser(
        description = "Change relative paths to absolute path."
    )
    # fmt: off
    parser.add_argument(
        "--json-file",
        required=True,
        help="path to the train json file"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="path to save filtered json output"
    )
    
    args = parser.parse_args()

    logger.info("Output file {}".format(args.output))

    obj = OrderedDict()
    obj = fix_path(obj, args.json_file)

    with open(args.output, 'w') as f:
        json.dump(obj, f, indent=4)

    logger.info("Dumped {} examples in {}".format(len(obj), args.output))

if __name__ == "__main__":
    main()
