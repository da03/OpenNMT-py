# Refinement

Throughout this file, we assume the working directory is `/raid/home/yuntian/edit/OpenNMT-py` on dgx-1.

### Dependencies

* `torch`: See [Pytorch.org](http://pytorch.org/)
* `torchtext`: `pip install torchtext`
* `torchvision`: `conda install torchvision`
* `Pillow`: `pip install Pillow`

### Command

1) Preprocess the data.

```
python preprocess.py -data_type img \
-src_dir edit -src_words_min_frequency 2 -tgt_words_min_frequency 2 \
-train_src edit/src-train.txt.errors -train_src_pred_img edit/src-train-pred-img.txt.errors -train_src_pred_text edit/tgt-train-pred.txt.errors -train_tgt edit/edits-train.txt.errors \
-valid_src edit/src-val.txt.errors -valid_src_pred_img edit/src-val-pred-img.txt.errors -valid_src_pred_text edit/tgt-val-pred.txt.errors -valid_tgt edit/edits-val.txt.errors \
-save_data demo
```

2) Train the model.

```
python train.py -model_type img  -data demo -save_model demo-model \
-gpuid 0 -batch_size 20 -max_grad_norm 20 -learning_rate 0.1
```

3) Translate the images.

```
python translate.py -data_type img -model demo-model_acc_80.78_ppl_2.54_e1.pt \
-src_dir edit -src edit/src-val.txt.errors -src_pred_img edit/src-val-pred-img.txt.errors -src_pred_text edit/tgt-val-pred.txt.errors \
-output pred.txt -verbose -gpu 0 -tgt edit/edits-val.txt.errors
```

Note that in the above command, `-tgt edit/edits-val.txt.errors` is optional, it's for evaluating ground truth labels' perplexities.

### Options

* `-src_dir`: The directory containing the images.

* `-train_src`: The file storing the orginal image paths. One path per line.
```
<image0_path>
<image1_path>
<image2_path>
...
```

* `-train_src_pred_img`: The file storing the predicted image paths (by a baseline model). One path per line.
```
<image0_path>
<image1_path>
<image2_path>
...
```

* `-train_src_pred_text`: The file storing the predicted tokens (by a baseline model). One label per line.
```
<label0_token0> <label0_token1> ... <label0_tokenN0>
<label1_token0> <label1_token1> ... <label1_tokenN1>
<label2_token0> <label2_token1> ... <label2_tokenN2>
...
```

A real example looks like
```
\left[ F ^ { \prime } - 2 \frac { K ^ { \prime } } { K } F \right] _ { z = z _ { i } ^ { + } } = 0 ,
```

* `-train_tgt`: The file storing the ground truth edit operations. One path per line.
```
<label0_token0> <label0_token1> ... <label0_tokenN0>
<label1_token0> <label1_token1> ... <label1_tokenN1>
<label2_token0> <label2_token1> ... <label2_tokenN2>
...
```

A real example looks like
```
_copy _copy _copy _copy _copy _ins_{ _copy _copy _copy _ins_} _copy _copy _copy _copy _copy _copy _copy _copy _del _copy _copy _copy _copy _copy _copy _copy _copy _copy _copy _copy _copy _copy _copy _ins_{ _copy _copy _copy _ins_} _copy _copy _copy _copy _copy _copy _copy _copy _del _copy _del _copy _copy _copy _copy _copy _copy _copy _copy _copy _copy _copy _copy _copy _del _ins_\; _copy _copy _copy _copy _copy _copy _copy _copy _ins_{ _copy _copy _copy _ins_} _copy _copy _copy _copy _copy _copy _copy _copy _copy _copy _copy _ins_{ _copy _copy _copy _ins_} _copy _copy _copy _copy _copy _copy _copy _copy _copy _copy _copy _del _ins_\; _copy
```
