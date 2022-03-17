#!/usr/bin/env python3
# This script is useful to make sure text files have text only in lower case
import argparse
import codecs

parser = argparse.ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("output_file")
args = parser.parse_args()

assert args.input_file != args.output_file, "Input and Output files cannot be the same"

with codecs.open(args.input_file, 'r', 'utf-8') as ipf, codecs.open(args.output_file, 'w', 'utf-8') as opf:
    for ln in ipf:
        lns = ln.strip().split()
        key = lns[0]
        new_ln = ' '.join([w.lower() for w in lns[1:]])
        opf.write('{} {}\n'.format(key, new_ln))
    opf.close()
