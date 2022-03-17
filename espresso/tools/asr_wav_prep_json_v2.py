#!/usr/bin/env python3
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import OrderedDict
import json
import logging
import sys
import subprocess
import io
from tqdm import tqdm
from multiprocessing import Pool

import soundfile as sf

import warnings
warnings.filterwarnings("ignore")


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("espresso.tools.asr_prep_json")


def wav2frames_multiprocess(obj):
    """ This function reads audio file and return data in numpy.float32 array.
        "lru_cache" holds recently loaded audio so that can be called
        many times on the same audio file.
        OPTIMIZE: controls lru_cache size for random access,
        considering memory size
    """
    if "duration" in obj and "sampling_rate" in obj:
        return round(obj['duration'] * obj['sampling_rate'])

    wav_rxfilename = obj['wav']
    wav_rxfilename = wav_rxfilename.strip()

    if wav_rxfilename.endswith('|'):
        # input piped command
        p = subprocess.Popen(wav_rxfilename[:-1], shell=True,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        frames = sf.info(io.BytesIO(p.stdout.read())).frames
    elif wav_rxfilename == '-':
        # stdin
        frames = sf.info(sys.stdin).frames
    else:
        # normal wav file
        frames = sf.info(wav_rxfilename).frames
    return frames


def get_frames_multiprocess(ordered_dict, key='utt2len'):
    values = ordered_dict.values()            #ordered keys and values
    p = Pool(3)
    res = list(tqdm(p.imap(wav2frames_multiprocess, values), total=len(values)))
    for i, (k, v) in enumerate(ordered_dict.items()):
        ordered_dict[k][key] = res[i]


def read_file(ordered_dict, key, dtype, *paths, allow_new_utt=False):
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                utt_id, val = line.strip().split(None, 1)
                if utt_id in ordered_dict:
                    assert key not in ordered_dict[utt_id], \
                        "Duplicate utterance id " + utt_id + " in " + key
                    ordered_dict[utt_id].update({key: dtype(val)})
                else:
                    if allow_new_utt:
                        ordered_dict[utt_id] = {key: val}
    return ordered_dict


def add_same_key(ordered_dict, key, val):
    for k, v in ordered_dict.items():
        ordered_dict[k][key] = val
    return ordered_dict


def main():
    parser = argparse.ArgumentParser(
        description="Wrap all related files of a dataset into a single json file"
    )
    # fmt: off
    parser.add_argument("--wav-files", nargs="+", required=True,
                        help="path(s) to the wav files")
    parser.add_argument("--token-text-files", nargs="+", default=None,
                        help="path(s) to token_text file(s)")
    parser.add_argument("--text-files", nargs="+", default=None,
                        help="path(s) to text file(s)")
    parser.add_argument("--numerator-fst-files", nargs="+", default=None,
                        help="path(s) to numerator fst file(s)")
    parser.add_argument("--alignment-files", nargs="+", default=None,
                        help="path(s) to alignment file(s)")
    parser.add_argument("--duration-files", nargs="+", default=None,
                        help="path(s) to duration file(s)")
    parser.add_argument("--sampling-rate", type=int, default=None,
                        help="sampling-rate")
    parser.add_argument("--output", required=True, type=argparse.FileType("w"),
                        help="path to save json output")
    args = parser.parse_args()
    # fmt: on

    obj = OrderedDict()
    obj = read_file(obj, "wav", str, *(args.wav_files), allow_new_utt=True)
    if args.token_text_files is not None:
        obj = read_file(obj, "token_text", str, *(args.token_text_files))
    if args.text_files is not None:
        obj = read_file(obj, "text", str, *(args.text_files))
    if args.numerator_fst_files is not None:
        obj = read_file(obj, "numerator_fst", str, *(args.numerator_fst_files))
    if args.alignment_files is not None:
        obj = read_file(obj, "alignment", str, *(args.alignment_files))
    if args.duration_files is not None:
        obj = read_file(obj, "duration", float, *(args.duration_files))
    if args.sampling_rate is not None:
        obj = add_same_key(obj, "sampling_rate", int(args.sampling_rate))

    get_frames_multiprocess(obj, "utt2len")
    json.dump(obj, args.output, indent=4)
    logger.info("Dumped {} examples in {}".format(len(obj), args.output.name))


if __name__ == "__main__":
    main()
