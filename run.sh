arr=(1 2 10)
for value in "${arr[@]}"
do
   echo "python train.py -data ../data/ar-en/data-feature -save_model ../data/ar-en/model-final-$value -gpuid 0 -train_from ../data/ar-en/model_acc_59.34_ppl_8.99_e13.pt -lamb $value >> log.train 2>&1"
   python train.py -data ../data/ar-en/data-feature -save_model ../data/ar-en/model-final-$value -gpuid 0 -train_from ../data/ar-en/model_acc_59.34_ppl_8.99_e13.pt -lamb $value >> log.train 2>&1
done
