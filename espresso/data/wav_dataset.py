#!/usr/bin/python

# Copyright (c) Apoorv Vyas
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import subprocess
import io
from typing import List

import numpy as np

import soundfile as sf

import torch

from fairseq.data import FairseqDataset
from fairseq.data.audio.raw_audio_dataset import RawAudioDataset

logger = logging.getLogger(__name__)


def load_wav(wav_rxfilename, start=0, end=None):
    """This function reads audio file and return data in numpy.float32 array.
    "lru_cache" holds recently loaded audio so that can be called
    many times on the same audio file.
    OPTIMIZE: controls lru_cache size for random access,
    considering memory size
    """
    wav_rxfilename = wav_rxfilename.strip()
    if wav_rxfilename.endswith("|"):
        # input piped command
        p = subprocess.Popen(
            wav_rxfilename[:-1],
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        data, samplerate = sf.read(io.BytesIO(p.stdout.read()), dtype="float32")
        # cannot seek
        data = data[start:end]
    elif wav_rxfilename == "-":
        # stdin
        data, samplerate = sf.read(sys.stdin, dtype="float32")
        # cannot seek
        data = data[start:end]
    else:
        # normal wav file
        data, samplerate = sf.read(wav_rxfilename, start=start, stop=end)
    return data, samplerate


class WavAudioDataset(RawAudioDataset):
    def __init__(
        self,
        utt_ids: List[str],
        wavs: List[str],
        utt2len: List[int],
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
        )
        assert len(utt_ids) == len(wavs)

        self.line_inds = set()
        self.sizes = []  # length of each utterance
        self.utt_ids = []
        self.wavs = []

        skipped = 0

        i = -1
        for utt_id, utt_len, utt_wav in zip(utt_ids, utt2len, wavs):
            i += 1
            sz = utt_len
            if min_sample_size is not None and sz < min_sample_size:
                skipped += 1
                continue
            self.line_inds.add(i)
            self.wavs.append(utt_wav)
            self.utt_ids.append(utt_id)
            self.sizes.append(sz)

        logger.info(f"loaded {len(self.wavs)}, skipped {skipped} samples")

    def __getitem__(self, index):
        import soundfile as sf

        fname = self.wavs[index]
        utt_id = self.utt_ids[index]
        # wav, curr_sample_rate = sf.read(fname)
        wav, curr_sample_rate = load_wav(fname)
        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)
        return {"id": index, "source": feats, "utt_id": utt_id}

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s["source"]) for s in samples]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(len(sources), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        if self.pad:
            input["padding_mask"] = padding_mask

        ids = torch.LongTensor([s["id"] for s in samples])
        utt_ids = [s["utt_id"] for s in samples]

        return {"id": ids, "net_input": input, "utt_ids": utt_ids, "input_sizes": sizes}

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::]
