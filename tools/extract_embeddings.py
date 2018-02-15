from __future__ import division

import torch
import argparse
import opts
import onmt
from onmt.Utils import use_gpu

from onmt.ModelConstructor import make_embeddings, \
                            make_encoder, make_decoder

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-output_dir', default='.',
                    help="""Path to output the embeddings""")
parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")


def write_embeddings(filename, dict, embeddings):
    with open(filename, 'wb') as file:
        for i in range(len(embeddings)):
            str = dict.itos[i].encode("utf-8")
            for j in range(len(embeddings[0])):
                str = str + (" %5f" % (embeddings[i][j])).encode("utf-8")
            file.write(str + ("\n").encode("utf-8"))


def main():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]
    opt = parser.parse_args()
    checkpoint = torch.load(opt.model)
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    model_opt = checkpoint['opt']
    src_dict = checkpoint['vocab'][1][1]
    tgt_dict = checkpoint['vocab'][0][1]
    feature_dicts = []

    # Add in default model arguments, possibly added since training.
    checkpoint = torch.load(opt.model,
                            map_location=lambda storage, loc: storage)
    fields = onmt.IO.load_fields(checkpoint['vocab'])

    model_opt = checkpoint['opt']
    for arg in dummy_opt.__dict__:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]

    _type = model_opt.encoder_type
    copy_attn = model_opt.copy_attn

    model = onmt.ModelConstructor.make_base_model(
                            model_opt, fields, use_gpu(opt), checkpoint)
    encoder = model.encoder
    decoder = model.decoder
    #embeddings = make_embeddings(model_opt, src_dict, feature_dicts)
    #encoder = make_encoder(model_opt, embeddings)

    #embeddings = make_embeddings(model_opt, tgt_dict, feature_dicts,
    #                             for_encoder=False)
    #decoder = make_decoder(model_opt, embeddings)

    ## Make NMTModel(= encoder + decoder).
    #model = NMTModel(encoder, decoder)

    encoder_embeddings = encoder.embeddings.word_lut.weight.data.tolist()
    decoder_embeddings = decoder.embeddings.word_lut.weight.data.tolist()

    print("Writing source embeddings")
    write_embeddings(opt.output_dir + "/src_embeddings.txt", src_dict,
                     encoder_embeddings)

    print("Writing target embeddings")
    write_embeddings(opt.output_dir + "/tgt_embeddings.txt", tgt_dict,
                     decoder_embeddings)

    print('... done.')
    print('Converting model...')


if __name__ == "__main__":
    main()
