"""fit.pipeline: the stages of ``python -m fit.build_db`` (raw tmQM/lnQM -> self-describing constraints DB).

Stages (each a function with explicit inputs and outputs, logged with wall time and peak RSS):
extract -> census/filter -> expand -> fit -> prune -> ref_ll -> bends -> assemble -> reproduce -> rescore.
All parallel work goes through one shared spawn pool (``pool.TaskPool``).
"""
