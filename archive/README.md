# Archived experiment implementations

`train_gpt_rpb_hybrid_geometry_pre_norfix.py` is the implementation used
for the original Stage 3 Nor-style sweep.

Its Nor adaptation was effectively inactive because the absolute RPB update
scale caused epsilon to dominate the row-wise second-moment normalization.
The results from that Stage 3 sweep should not be interpreted as a valid
NorMuon-style ablation.

The active corrected implementation is:

    ../train_gpt_rpb_hybrid_geometry.py
