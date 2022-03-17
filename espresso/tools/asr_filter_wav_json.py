#!/usr/bin/env python3

from collections import OrderedDict
import json
import argparse
import sys
import re
import logging
from tqdm import tqdm

try:
    from pychain import ChainGraph
    import simplefst
except ImportError:
    raise ImportError("Please install OpenFST and PyChain by `make openfst pychain` after entering espresso/tools")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("espresso.tools.asr_filter_small")


def parse_rxfile(rxfile):
    # separate offset from filename
    m = re.match(r"(\S+):([0-9]+)", rxfile)
    assert m is not None, "Illegal rxfile: {}".format(rxfile)
    return m.group(1), int(m.group(2))


def filter_utts(obj, data_json_path, min_length=None, max_length=None):
    with open(data_json_path, "rb") as f:
        loaded_json = json.load(f, object_pairs_hook=OrderedDict)
    
    logger.info("Total utterance originally: {}".format(len(loaded_json)))

    n_short = 0
    n_long = 0
    n_missing_states = 0
    for utt_id, val in tqdm(loaded_json.items()):
        assert "utt2len" in val, "utt2len key does not exist in {}".format(utt_id)
        assert utt_id not in obj, "Duplicate utterance found"
        add_this_utt = True
        if max_length and val["utt2len"] > max_length: 
            add_this_utt = False
            n_long += 1

        if min_length and val["utt2len"] < min_length: 
            add_this_utt = False
            n_short += 1
        
        if "numerator_fst" in val:
            rxfile = val["numerator_fst"]
            file_path, offset = parse_rxfile(rxfile)
            fst = simplefst.StdVectorFst.read_ark(file_path, offset)
            if fst.num_states() == 0:
                add_this_utt = False
                n_missing_states += 1
        
        if add_this_utt:
            obj[utt_id] = val

    return obj, len(loaded_json) - len(obj), n_long, n_short, n_missing_states


def main():
    parser = argparse.ArgumentParser(
        description = "Filter wav inputs in training set."
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
    parser.add_argument(
        "--min-length",
        default=0,
        type=int,
        help="min length to filter in frames"
    )
    parser.add_argument(
        "--max-length",
        default=None,
        type=int,
        help="max length to filter in frames"
    )

    args = parser.parse_args()

    logger.info("Output file {}".format(args.output))

    obj = OrderedDict()
    obj, n_filtered, n_long, n_short, n_missing_states = filter_utts(
        obj,
        args.json_file,
        args.min_length,
        args.max_length
    )

    logger.info("Total utterance after filtering: {}".format(len(obj)))
    logger.info("Total utterances filtered: {}".format(n_filtered))
    logger.info("Long utterances filtered: {}".format(n_long))
    logger.info("Short utterances filtered: {}".format(n_short))
    logger.info("Missing FST states utterances filtered: {}".format(n_missing_states))

    with open(args.output, 'w') as f:
        json.dump(obj, f, indent=4)

    logger.info("Dumped {} examples in {}".format(len(obj), args.output))

if __name__ == "__main__":
    main()
