"""HumanScore scoring: the six plausibility metrics of a reconstructed session.

The last pipeline stage.  It reads what the reconstruction engines wrote into a session
directory (keypoints, skeleton fit, body mesh, HADM detections) and produces six 0-100
scores (higher = more plausible), three tier scores and an overall score.

* :mod:`hmb.scoring.score_session` scores a session (``python -m hmb.scoring.score_session``)
  and writes ``<S>/scores.json``;
* :mod:`hmb.scoring.extra_limbs`, :mod:`~hmb.scoring.bone_length`, :mod:`~hmb.scoring.joint_range`
  and :mod:`~hmb.scoring.self_collision` are the metric modules (each documents its definition
  and constants, and can be run on its own);
* :mod:`hmb.scoring.paper_scores` runs the original paper implementation (``metrics/``),
  used for kinematic extremes and motion smoothness and, on request, for all six;
* :mod:`hmb.scoring.aggregate` holds the shared frequency / severity / persistence aggregation.
"""
