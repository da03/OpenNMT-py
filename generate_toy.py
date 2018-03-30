import os, sys, random, hashlib, shutil, time, re
import multiprocessing
from multiprocessing import Pool
from subprocess import call
import numpy as np
import math
seed = 939270

NUM_THREADS = 128

DEVNULL = open(os.devnull, "w")

# replace \pmatrix with \begin{pmatrix}\end{pmatrix}
# replace \matrix with \begin{matrix}\end{matrix}
template = r"""
\documentclass[12pt]{article}
\pagestyle{empty}
\usepackage{amsmath}
\newcommand{\mymatrix}[1]{\begin{matrix}#1\end{matrix}}
\newcommand{\mypmatrix}[1]{\begin{pmatrix}#1\end{pmatrix}}
\begin{document}
\begin{displaymath}
%s
\end{displaymath}
\end{document}
"""

def worker_main(queue):
    while True:
        item = queue.get(True)
        try:
            full_path, label = item
            path = os.path.basename(full_path)
            l = label
            l = l.strip()
            l = l.replace(r'\pmatrix', r'\mypmatrix')
            l = l.replace(r'\matrix', r'\mymatrix')
            # remove leading comments
            l = l.strip('%')
            if len(l) == 0:
                l = '\\hspace{1cm}'
            # \hspace {1 . 5 cm} -> \hspace {1.5cm}
            for space in ["hspace", "vspace"]:
                match = re.finditer(space + " {(.*?)}", l)
                if match:
                    new_l = ""
                    last = 0
                    for m in match:
                        new_l = new_l + l[last:m.start(1)] + m.group(1).replace(" ", "")
                        last = m.end(1)
                    new_l = new_l + l[last:]
                    l = new_l 
            latex = l
            tex_path = path[:-4] + '.tex'
            with open(tex_path, 'w') as fout:
                fout.write(template%latex)
            call(["pdflatex", '-interaction=nonstopmode', '-halt-on-error', tex_path],
                            stdout=DEVNULL, stderr=DEVNULL)
            call(["convert", "-density", "200", "-quality", "100", path[:-4]+".pdf", path[:-4]+".png"],
                            stdout=DEVNULL, stderr=DEVNULL)
            if os.path.exists(tex_path):
                os.remove(tex_path)
            pdf_path = path[:-4] + '.pdf'
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            pdf_path = path[:-4] + '.log'
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            pdf_path = path[:-4] + '.aux'
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            if os.path.exists(path[:-4]+'.png'):
                shutil.move(path[:-4]+'.png', full_path)
        except Exception as e:
            try:
                if os.path.exists(tex_path):
                    os.remove(tex_path)
                pdf_path = path[:-4] + '.pdf'
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                pdf_path = path[:-4] + '.log'
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                pdf_path = path[:-4] + '.aux'
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                if os.path.exists(path[:-4]+'.png'):
                    shutil.move(path[:-4]+'.png', full_path)
            except Exception as e:
                pass

def render_corrupt_prob(labels, output_data_path, output_label_path, output_img_dir, output_dir):
    finished = set([])
    with open(output_data_path, 'w') as fdata:
        with open(output_label_path, 'w') as flabel:
            num_written = 0
            for label_id,label in enumerate(labels):
                if label_id % 1000 == 0:
                    print (label_id)
                name = hashlib.sha1(label.encode('utf-8')).hexdigest()
                name = os.path.join(output_img_dir, name)
                flabel.write(label+'\n')
                fdata.write('%s.png\n'%(os.path.relpath(name, output_dir)))
                if name not in finished:
                    the_queue.put((name+'.png', label))
                    finished.add(name)
                num_written += 1

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print >> sys.stderr, 'Usage: python %s <label-path-tok> <output-dir>'%sys.argv[0]
        sys.exit(1)

    label_path = sys.argv[1]
    output_dir = sys.argv[2]

    labels = [label.strip() for label in open(label_path).readlines()]
    labels = labels

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    the_queue = multiprocessing.Queue()
    the_pool = multiprocessing.Pool(NUM_THREADS, worker_main, (the_queue,))
    random.seed(seed)
    np.random.seed(seed)
    output_ind_dir = os.path.join(output_dir, 'main')
    if not os.path.exists(output_ind_dir):
        os.makedirs(output_ind_dir)
    output_data_path = os.path.join(output_ind_dir, 'data.txt')
    output_label_path = os.path.join(output_ind_dir, 'labels.txt')
    output_img_dir = os.path.join(output_ind_dir, 'images')
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    render_corrupt_prob(labels, output_data_path, output_label_path, output_img_dir, output_dir)
     
    while True:
        qsize = the_queue.qsize()
        time.sleep(0.5)
    time.sleep(100)
