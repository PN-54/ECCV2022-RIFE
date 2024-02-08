import logging
import time
from multiprocessing.pool import ThreadPool
from itertools import islice
from math import ceil, log10
import os

import numcodecs
from imagecodecs.numcodecs import Jpegxl

numcodecs.register_codec(Jpegxl)
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
)


def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def progress(iterable, *, description, total=None):
    if total is None:
        total = len(iterable)
    w = ceil(log10(total))
    tbegin = time.time()
    t0 = tbegin
    for i, x in enumerate(iter(iterable)):
        yield x
        t1 = time.time()
        if t1 - t0 > 1:  # No more than 1 log message per second
            r = (i + 1) / (t1 - tbegin)
            if r >= 1:
                rate = f"{r:.0f} iter/s"
            else:
                rate = f"{1/r:.0f} s/iter"
            count = f"{str(i).rjust(w)}/{total}"
            eta = round((total - i) / r)
            eta = f"{eta//3600:02}:{(eta % 3600)//60:02}:{eta%60:02}"
            logging.info(
                f"[{count}] (rate: {rate}, remaining: {eta}) {description}"
            )
            t0 = time.time()
    logging.info(f"[{total}/{total}] (average rate: {r:.0f}/s) {description}. DONE")


def parallel_load(array, *, chunksize=1, workers=os.cpu_count()):
    def load(i):
        return array[i]
    with ThreadPool(workers) as pool:
        for x in pool.imap_unordered(load, range(len(array)), chunksize):
            yield x
