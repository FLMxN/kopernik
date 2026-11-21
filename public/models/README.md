# ONNX models

FP16 ONNX exports of the two ResNet-50 classifiers from
[`flmxn/resnet50-streetview`](https://huggingface.co/flmxn/resnet50-streetview).

- `country_fp16.onnx` — 56 country classes;
- `region_fp16.onnx` — 7 region classes.

Class order is the checkpoint `label_mapping` (alphabetical ISO order for
countries). The upstream model card marks the weights as GPL; the Kopernik
code is MIT. See the upstream model repository for the dataset credit and
license details.
