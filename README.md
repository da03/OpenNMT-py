# OpenNMT-py adapted for Single Task Neural Machine Translation

## Usage:

### Preprocess data with source side features

1. The target file is the same, one tokenized sentence per line. The source file should provide the tags (POS, Morphology) with the following format:

```
word1￨pos1￨morphology1 word2￨pos2￨morphology2 word3￨pos3￨morphology3 ...
```

Below is a real snippet taken from source validation data:

```
هل￨RP￨INTERROG_PART تعرفون￨VBP￨IV2MP+IV+IVSUFF_SUBJ:MP_MOOD:I أن￨IN￨SUB_CONJ أحد￨CD￨NOUN_NUM+CASE_DEF_ACC المتع￨DT+NN￨DET+NOUN+CASE_DEF_GEN الكبيرة￨DT+JJ￨DET+ADJ+NSUFF_FEM_SG+CASE_DEF_GEN للسفر￨DT+NN￨PREP+DET+NOUN+CASE_DEF_GEN وأحد￨CD￨NOUN_NUM+CASE_INDEF_GEN مباهج￨NN￨NOUN+CASE_INDEF_GEN أبحاث￨NN￨NOUN+CASE_INDEF_GEN الإثنوجرافيا￨DT+NN￨DET+NOUN في￨IN￨PREP فرصة￨NN￨NOUN+NSUFF_FEM_SG+CASE_DEF_NOM العيش￨DT+NN￨DET+NOUN+CASE_DEF_GEN بين￨NN￨NOUN+CASE_DEF_ACC أولئك￨NN￨NOUN الذين￨WP￨REL_PRON لم￨RP￨NEG_PART ينسوا￨VBP￨IV3MP+IV+IVSUFF_SUBJ:MP_MOOD:SJ الأساليب￨DT+NN￨DET+NOUN+CASE_DEF_ACC القديمة￨DT+JJ￨DET+ADJ+NSUFF_FEM_SG+CASE_DEF_GEN الذين￨WP￨REL_PRON لا￨RP￨NEG_PART زالوا￨VBD￨PV+PVSUFF_SUBJ:3MP يشعرون￨VBP￨IV3MP+IV+IVSUFF_SUBJ:MP_MOOD:I بماضيهم￨JJ￨PREP+ADJ+NSUFF_MASC_PL_ACC+POSS_PRON_3MP في￨IN￨PREP الرياح￨DT+NN￨DET+NOUN+CASE_DEF_GEN ويلمسونه￨VBP￨CONJ+IV3MP+IV+IVSUFF_SUBJ:MP_MOOD:I+IVSUFF_DO:3MS في￨IN￨PREP الأحجار￨DT+NN￨DET+NOUN+CASE_DEF_GEN التي￨WP￨REL_PRON صقلتها￨VBD￨PV+PVSUFF_SUBJ:3FS+PVSUFF_DO:3FS الأمطار￨DT+NN￨DET+NOUN+CASE_DEF_NOM ويتذوقونه￨VBP￨CONJ+IV3MP+IV+IVSUFF_SUBJ:MP_MOOD:I+IVSUFF_DO:3MS في￨IN￨PREP أوراق￨NN￨NOUN+CASE_DEF_GEN النباتات￨DT+NNS￨DET+NOUN+NSUFF_FEM_PL+CASE_DEF_GEN المرة￨DT+NN￨DET+NOUN+NSUFF_FEM_SG+CASE_DEF_GEN
```

2. Generate data for OpenNMT by

```
python preprocess.py -train_src ar-en/src-train-features.txt -train_tgt ar-en/tgt-train.txt -valid_src ar-en/src-val-features.txt -valid_tgt ar-en/tgt-val.txt -save_data ar-en/data-feature
```

### Train with joint objective: machine translation + fooling discriminator

At this stage, for each batch, we optimize the following two objectives:

```
loss1 = losstranslation - lossdiscriminator * lamb
```

For `loss1`, we are training our model (w/o training discriminator) such that we can both get good translation and fool the discriminator.

```
loss2 = lossdiscriminator
```

For `loss2`, we are training our discriminator (other parts are untouched) such that it can classify the tag based on embeddings or intermediate layer outputs.


Our training command looks like below:

```
python train.py -data ar-en/data-feature -save_model ar-en/model -phase 1 -classify_layer 2 -num_classifiers 5 -feat_id 0 -lamb 0.01 -gpuid 0
```

There are several important options:

* phase: should be set to 1 for this joint objective
* classify_layer: the layer that we want to classify, layer 0 is word embedding, layer 1 is the first layer output of encoder rnn
* num_classifiers: how many classifiers to use for this phase, note that we use multiple classifiers to enable training out tag information better, but wheter this is useful is yet to be examined
* feat_id: the tag that we use. Note that for source file with features separated by ￨, the first tag after word is indexed 0, and the second is indexed as 1. For our example, 0 corresponds to POS, 1 corresponds to Morphology.
* lamb: the tradeoff between translation loss and discriminator loss.
* train_from: optional, use it only if we want to use a pretrained model.


### Fix the translation parts of the model, train a new discriminator

We then fix the translation parts and train a new discriminator to see how good information has been trained out.

```
python train.py -data ar-en/data-feature -save_model ar-en/model -phase 2 -classify_layer 2 -feat_id 0 -gpuid 0
```

* phase: should be set to 2 for this joint objective

### Translate to target and classify source tags

Then hand-pick a model, run the below translation command, it will produce both target and source tags (source tags are of the same format as source file with features mentioned above).

```
python translate.py -src ar-en/src-val-features.txt -model ar-en/model-phase2_acc_56.66_93.32_ppl_10.97_1.21_e11.pt -output ar-en/pred.txt -verbose
```

### Evaluation:

* `tools/multi-bleu.perl` provides BLEU evaluation.
* `tools/extract_embeddings.py` can convert extract word embeddings.
