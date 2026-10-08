"""Track 3: defect detection + localisation pipeline.

Image -> preprocessing -> YOLO detection -> crop -> CNN classification
      -> engineer review -> evaluation -> automated report

Modules:
  data      dataset converters (DeepPCB, MVTec AD masks, any YOLO export), splits, stats
  preprocess  the one place that turns a raw image (+ optional reference) into model input
  boxes     box utilities (IoU, matching, crops)
  evaluate  detection metrics: P/R/F1, AP/mAP, confusion incl. background, FP/FN lists
  viz       overlays, galleries, PR curves
  report    HTML report
  review    engineer review page + ingesting decisions
"""
