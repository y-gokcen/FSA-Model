#!/usr/bin/env python3
"""Summarize the CSV output of a set of FSA runs.

Usage: python3 tools/summarize.py <dir-with-fsa_*_runNN_test.csv>

Also reads the *_test_10k.csv files written by -test-only runs (epochs=-1 there).

For every run: epochs trained, fraction of valid predictions over all 10k test
trials (a possible next token), and fraction of correct predictions (H vs I)
on the checked transitions after C/D/E, overall, per branch (A = state 5,
B = state 6) and by the number of G tokens that preceded the C/D/E.
"""
import csv, collections, glob, os, sys

d = sys.argv[1]
# *_test.csv from training runs; *_test_10k.csv from -test-only runs
tests = sorted(glob.glob(os.path.join(d, 'fsa_*_run??_test.csv')) +
               glob.glob(os.path.join(d, 'fsa_*_run??_test_10k.csv')))
allv = []; allc = []
for tf in tests:
    trf = tf.replace('_test_10k.csv', '.csv').replace('_test.csv', '.csv')
    try:
        ep = max(int(r['Epoch']) for r in csv.DictReader(open(trf))) + 1
    except Exception:
        ep = -1
    t = list(csv.DictReader(open(tf)))
    valid = sum(int(r['Valid']) for r in t) / len(t)
    byb = collections.defaultdict(lambda: [0, 0]); byg = collections.defaultdict(lambda: [0, 0]); ng = 0
    for r in t:
        if r['Stim'] in 'AB': ng = 0
        elif r['Stim'] == 'G': ng += 1
        if r['StateNode'] in ('5', '6'):
            ok = r['Predicted'] == r['NextStim']
            byb[r['StateNode']][0] += ok; byb[r['StateNode']][1] += 1
            byg[min(ng, 4)][0] += ok; byg[min(ng, 4)][1] += 1
    ca = sum(v[0] for v in byb.values()) / max(1, sum(v[1] for v in byb.values()))
    allv.append(valid); allc.append(ca)
    print("%s epochs=%4d valid=%.3f correct=%.3f (A %.2f B %.2f) byG: %s" % (
        os.path.basename(tf).replace('_test_10k.csv', '').replace('_test.csv', ''), ep, valid, ca,
        byb['5'][0] / max(1, byb['5'][1]), byb['6'][0] / max(1, byb['6'][1]),
        " ".join("%d:%.2f" % (g, byg[g][0] / byg[g][1]) for g in sorted(byg))))
if allv:
    print("MEAN valid=%.3f correct=%.3f  solved(>=0.99 correct)=%d/%d" % (
        sum(allv) / len(allv), sum(allc) / len(allc), sum(c >= 0.99 for c in allc), len(allc)))
