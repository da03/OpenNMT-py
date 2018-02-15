import os, sys

config = {}
config['dataset'] = ['ar-en']
config['feature'] = ['morph', 'pos']
config['train_from'] = ['none', 'pretrained']
config['lamb'] = [0.1]
config['layer'] = [0]


for dataset in config['dataset']:
    for feature in config['feature']:
        for layer in config['layer']:
            # random
            options = ' -gpuid 0 '
            if feature == 'pos':
                options += ' -feat_id 0 '
            elif feature == 'morph':
                options += ' -feat_id 1 '
            options += ' -phase 2 '
            cmd = 'python train.py -data ../data/%s/data-feature -save_model ../data/%s/model-final-random-%s %s'%(dataset, dataset, '_'.join([str(item) for item in [dataset,feature,layer]]).replace(' ','_'), options)
            print (cmd)
            sys.stdout.flush()
            os.system(cmd)
            for train_from in config['train_from']:
                for lamb in config['lamb']:
                    # first, train translation with adversarial loss
                    options = ' -gpuid 0 '
                    if feature == 'pos':
                        options += ' -feat_id 0 '
                    elif feature == 'morph':
                        options += ' -feat_id 1 '
                    if train_from == 'pretrained':
                        options += ' -train_from ../data/ar-en/model_acc_59.34_ppl_8.99_e13.pt '
                    options += ' -phase 1 '
                    options += ' -lamb %f '%lamb
                    cmd = 'python train.py -data ../data/%s/data-feature -save_model ../data/%s/model-final-phase1-%s %s'%(dataset, dataset, '_'.join([str(item) for item in [dataset,feature,train_from,lamb,layer]]).replace(' ','_'), options)
                    print (cmd)
                    sys.stdout.flush()
                    os.system(cmd)
                    # then, test bleu
                    # lastly, train classifier and test accuracy (use best validation accuracy)
                    options = ' -gpuid 0 '
                    options += ' -phase 2 '
