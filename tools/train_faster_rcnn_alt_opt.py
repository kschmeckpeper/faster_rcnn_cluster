#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Faster R-CNN network using alternating optimization.
This tool implements the alternating optimization algorithm described in our
NIPS 2015 paper ("Faster R-CNN: Towards Real-time Object Detection with Region
Proposal Networks." Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.)
"""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from rpn.generate import imdb_proposals
import argparse
import pprint
import numpy as np
import sys, os
from os import listdir
from os.path import isfile, join
import multiprocessing as mp
import cPickle
import shutil

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--net_name', dest='net_name',
                        help='network name (e.g., "ZF")',
                        default=None, type=str)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_roidb(imdb_name, rpn_file=None):
    imdb = get_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
    if rpn_file is not None:
        imdb.config['rpn_file'] = rpn_file
    roidb = get_training_roidb(imdb)
    return roidb, imdb

def get_solvers(net_name):
    # Faster R-CNN Alternating Optimization
    n = 'faster_rcnn_alt_opt'
    # Solver for each training stage
    solvers = [[net_name, n, 'stage1_rpn_solver60k80k.pt'],
               [net_name, n, 'stage1_fast_rcnn_solver30k40k.pt'],
               [net_name, n, 'stage2_rpn_solver60k80k.pt'],
               [net_name, n, 'stage2_fast_rcnn_solver30k40k.pt']]
    solvers = [os.path.join(cfg.MODELS_DIR, *s) for s in solvers]
    # Iterations for each training stage
    max_iters = np.array([80000, 40000, 80000, 40000])
    # max_iters = [100, 100, 100, 100]
    # Test prototxt for the RPN
    rpn_test_prototxt = os.path.join(
        cfg.MODELS_DIR, net_name, n, 'rpn_test.pt')
    return solvers, max_iters, rpn_test_prototxt

# ------------------------------------------------------------------------------
# Pycaffe doesn't reliably free GPU memory when instantiated nets are discarded
# (e.g. "del net" in Python code). To work around this issue, each training
# stage is executed in a separate process using multiprocessing.Process.
# ------------------------------------------------------------------------------

def _init_caffe(cfg):
    """Initialize pycaffe in a training process.
    """

    import caffe
    # fix the random seeds (numpy and caffe) for reproducibility
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)
    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)

def train_rpn(queue=None, imdb_name=None, init_model=None, solver=None,
              starting_iters=0, max_iters=None, cfg=None):
    """Train a Region Proposal Network in a separate training process.
    """

    # Not using any proposals, just ground-truth boxes
    cfg.TRAIN.HAS_RPN = True
    cfg.TRAIN.BBOX_REG = False  # applies only to Fast R-CNN bbox regression
    cfg.TRAIN.PROPOSAL_METHOD = 'gt'
    cfg.TRAIN.IMS_PER_BATCH = 1
    print 'Init model: {}'.format(init_model)
    print('Using config:')
    pprint.pprint(cfg)

    import caffe
    _init_caffe(cfg)

    roidb, imdb = get_roidb(imdb_name)
    print 'roidb len: {}'.format(len(roidb))
    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    model_paths = train_net(solver, roidb, output_dir,
                            pretrained_model=init_model,
                            starting_iters=starting_iters,
                            max_iters=max_iters)
    # Cleanup all but the final model
    for i in model_paths[:-1]:
        os.remove(i)
    rpn_model_path = model_paths[-1]
    # Send final model path through the multiprocessing queue
    queue.put({'model_path': rpn_model_path})

def rpn_generate(queue=None, imdb_name=None, rpn_model_path=None, cfg=None,
                 rpn_test_prototxt=None):
    """Use a trained RPN to generate proposals.
    """

    cfg.TEST.RPN_PRE_NMS_TOP_N = -1     # no pre NMS filtering
    cfg.TEST.RPN_POST_NMS_TOP_N = 2000  # limit top boxes after NMS
    print 'RPN model: {}'.format(rpn_model_path)
    print('Using config:')
    pprint.pprint(cfg)

    import caffe
    _init_caffe(cfg)

    # NOTE: the matlab implementation computes proposals on flipped images, too.
    # We compute them on the image once and then flip the already computed
    # proposals. This might cause a minor loss in mAP (less proposal jittering).
    imdb = get_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for proposal generation'.format(imdb.name)

    # Load RPN and configure output directory
    rpn_net = caffe.Net(rpn_test_prototxt, rpn_model_path, caffe.TEST)
    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)
    # Generate proposals on the imdb
    rpn_proposals = imdb_proposals(rpn_net, imdb)
    # Write proposals to disk and send the proposal file path through the
    # multiprocessing queue
    rpn_net_name = os.path.splitext(os.path.basename(rpn_model_path))[0]
    rpn_proposals_path = os.path.join(
        output_dir, rpn_net_name + '_proposals.pkl')
    with open(rpn_proposals_path, 'wb') as f:
        cPickle.dump(rpn_proposals, f, cPickle.HIGHEST_PROTOCOL)
    print 'Wrote RPN proposals to {}'.format(rpn_proposals_path)
    queue.put({'proposal_path': rpn_proposals_path})

def train_fast_rcnn(queue=None, imdb_name=None, init_model=None, solver=None,
                    starting_iters=0, max_iters=None, cfg=None, rpn_file=None):
    """Train a Fast R-CNN using proposals generated by an RPN.
    """

    cfg.TRAIN.HAS_RPN = False           # not generating prosals on-the-fly
    cfg.TRAIN.PROPOSAL_METHOD = 'rpn'   # use pre-computed RPN proposals instead
    cfg.TRAIN.IMS_PER_BATCH = 2
    print 'Init model: {}'.format(init_model)
    print 'RPN proposals: {}'.format(rpn_file)
    print('Using config:')
    pprint.pprint(cfg)

    import caffe
    _init_caffe(cfg)

    roidb, imdb = get_roidb(imdb_name, rpn_file=rpn_file)
    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)
    # Train Fast R-CNN
    model_paths = train_net(solver, roidb, output_dir,
                            pretrained_model=init_model,
                            starting_iters=starting_iters,
                            max_iters=max_iters)
    # Cleanup all but the final model
    for i in model_paths[:-1]:
        os.remove(i)
    fast_rcnn_model_path = model_paths[-1]
    # Send Fast R-CNN model path over the multiprocessing queue
    queue.put({'model_path': fast_rcnn_model_path})

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        print "Loading cfg from file:", args.cfg_file
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        print "Loading cfg from list:", args.set_cfgs
        cfg_from_list(args.set_cfgs)
    cfg.GPU_ID = args.gpu_id

    # --------------------------------------------------------------------------
    # Pycaffe doesn't reliably free GPU memory when instantiated nets are
    # discarded (e.g. "del net" in Python code). To work around this issue, each
    # training stage is executed in a separate process using
    # multiprocessing.Process.
    # --------------------------------------------------------------------------

    # queue for communicated results between processes
    mp_queue = mp.Queue()
    # solves, iters, etc. for each training stage
    solvers, max_iters, rpn_test_prototxt = get_solvers(args.net_name)
    print "Created solvers"

    imdb = get_imdb(args.imdb_name)
    print "Got imdb from:", args.imdb_name
    output_dir = get_output_dir(imdb)
    print "Got output_dir:", output_dir
    prev_saved_models = [f for f in listdir(output_dir) if isfile(join(output_dir, f))]
    

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 1 RPN, init from ImageNet model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    prev_rpn_stage_1 = [f for f in prev_saved_models if "rpn_stage1" in f and f[-11:] == '.caffemodel']

    pretrained_model = args.pretrained_model
    starting_iters = 0
    if len(prev_rpn_stage_1) > 0:
        num_iters = [int(s.replace('_', '.').split('.')[-2]) for s in prev_rpn_stage_1]

        latest_index = np.argmax(num_iters)

        starting_iters = num_iters[latest_index]
        pretrained_model = join(output_dir, prev_rpn_stage_1[latest_index])


    if starting_iters < max_iters[0]:
        cfg.TRAIN.SNAPSHOT_INFIX = 'stage1'
        mp_kwargs = dict(
                queue=mp_queue,
                imdb_name=args.imdb_name,
                init_model=pretrained_model,
                solver=solvers[0],
                starting_iters=starting_iters,
                max_iters=max_iters[0],
                cfg=cfg)
        p = mp.Process(target=train_rpn, kwargs=mp_kwargs)
        p.start()
        rpn_stage1_out = mp_queue.get()
        p.join()
    else:
        rpn_stage1_out = dict()
        rpn_stage1_out['model_path'] = pretrained_model


    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 1 RPN, generate proposals'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    prev_proposals_stage_1 = [f for f in prev_saved_models if "rpn_stage1" in f and f[-14:] == '_proposals.pkl']

    found_prev_proposal = False
    if len(prev_proposals_stage_1) > 0:
        for prev_proposal in prev_proposals_stage_1:
            if str(starting_iters) in prev_proposal:
                rpn_stage1_out['proposal_path'] = join(output_dir, prev_proposal)
                found_prev_proposal = True
                break


    if not found_prev_proposal:
        mp_kwargs = dict(
                queue=mp_queue,
                imdb_name=args.imdb_name,
                rpn_model_path=str(rpn_stage1_out['model_path']),
                cfg=cfg,
                rpn_test_prototxt=rpn_test_prototxt)
        p = mp.Process(target=rpn_generate, kwargs=mp_kwargs)
        p.start()
        rpn_stage1_out['proposal_path'] = mp_queue.get()['proposal_path']
        p.join()


    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 1 Fast R-CNN using RPN proposals, init from ImageNet model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    prev_rcnn_stage_1 = [f for f in prev_saved_models if "fast_rcnn_stage1" in f and f[-11:] == '.caffemodel']

    pretrained_model = args.pretrained_model
    starting_iters = 0
    if len(prev_rcnn_stage_1) > 0:
        num_iters = [int(s.replace('_', '.').split('.')[-2]) for s in prev_rcnn_stage_1]

        latest_index = np.argmax(num_iters)

        starting_iters = num_iters[latest_index]
        pretrained_model = join(output_dir, prev_rcnn_stage_1[latest_index])


    if starting_iters < max_iters[1]:
        cfg.TRAIN.SNAPSHOT_INFIX = 'stage1'
        mp_kwargs = dict(
                queue=mp_queue,
                imdb_name=args.imdb_name,
                init_model=pretrained_model,
                solver=solvers[1],
                starting_iters=starting_iters,
                max_iters=max_iters[1],
                cfg=cfg,
                rpn_file=rpn_stage1_out['proposal_path'])
        p = mp.Process(target=train_fast_rcnn, kwargs=mp_kwargs)
        p.start()
        fast_rcnn_stage1_out = mp_queue.get()
        p.join()
    else:
        fast_rcnn_stage1_out = dict()
        fast_rcnn_stage1_out['model_path'] = pretrained_model


    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 2 RPN, init from stage 1 Fast R-CNN model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'


    prev_rpn_stage_2 = [f for f in prev_saved_models if "rpn_stage2" in f and f[-11:] == '.caffemodel']

    pretrained_model = args.pretrained_model
    starting_iters = 0
    if len(prev_rpn_stage_2) > 0:
        num_iters = [int(s.replace('_', '.').split('.')[-2]) for s in prev_rpn_stage_2]

        latest_index = np.argmax(num_iters)

        starting_iters = num_iters[latest_index]
        pretrained_model = join(output_dir, prev_rpn_stage_2[latest_index])


    if starting_iters < max_iters[2]:
        cfg.TRAIN.SNAPSHOT_INFIX = 'stage2'
        mp_kwargs = dict(
                queue=mp_queue,
                imdb_name=args.imdb_name,
                init_model=str(fast_rcnn_stage1_out['model_path']),
                solver=solvers[2],
                starting_iters=starting_iters,
                max_iters=max_iters[2],
                cfg=cfg)
        p = mp.Process(target=train_rpn, kwargs=mp_kwargs)
        p.start()
        rpn_stage2_out = mp_queue.get()
        p.join()
    else:
        rpn_stage2_out = dict()
        rpn_stage2_out['model_path'] = pretrained_model



    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 2 RPN, generate proposals'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'


    prev_proposals_stage_2 = [f for f in prev_saved_models if "rpn_stage2" in f and f[-14:] == '_proposals.pkl']

    found_prev_proposal = False
    if len(prev_proposals_stage_2) > 0:
        for prev_proposal in prev_proposals_stage_2:
            if str(starting_iters) in prev_proposal:
                rpn_stage2_out['proposal_path'] = join(output_dir, prev_proposal)
                found_prev_proposal = True
                break


    if not found_prev_proposal:
        mp_kwargs = dict(
                queue=mp_queue,
                imdb_name=args.imdb_name,
                rpn_model_path=str(rpn_stage2_out['model_path']),
                cfg=cfg,
                rpn_test_prototxt=rpn_test_prototxt)
        p = mp.Process(target=rpn_generate, kwargs=mp_kwargs)
        p.start()
        rpn_stage2_out['proposal_path'] = mp_queue.get()['proposal_path']
        p.join()



    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 2 Fast R-CNN, init from stage 2 RPN R-CNN model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'


    prev_rcnn_stage_2 = [f for f in prev_saved_models if "fast_rcnn_stage2" in f and f[-11:] == '.caffemodel']

    pretrained_model = args.pretrained_model
    starting_iters = 0
    if len(prev_rcnn_stage_2) > 0:
        num_iters = [int(s.replace('_', '.').split('.')[-2]) for s in prev_rcnn_stage_2]

        latest_index = np.argmax(num_iters)

        starting_iters = num_iters[latest_index]
        pretrained_model = join(output_dir, prev_rcnn_stage_2[latest_index])


    if starting_iters < max_iters[3]:
        cfg.TRAIN.SNAPSHOT_INFIX = 'stage2'
        mp_kwargs = dict(
                queue=mp_queue,
                imdb_name=args.imdb_name,
                init_model=str(rpn_stage2_out['model_path']),
                solver=solvers[3],
                starting_iters=starting_iters,
                max_iters=max_iters[3],
                cfg=cfg,
                rpn_file=rpn_stage2_out['proposal_path'])
        p = mp.Process(target=train_fast_rcnn, kwargs=mp_kwargs)
        p.start()
        fast_rcnn_stage2_out = mp_queue.get()
        p.join()
    else:
        fast_rcnn_stage2_out = dict()
        fast_rcnn_stage2_out['model_path'] = pretrained_model

    # Create final model (just a copy of the last stage)
    final_path = os.path.join(
            os.path.dirname(fast_rcnn_stage2_out['model_path']),
            args.net_name + '_faster_rcnn_final.caffemodel')
    print 'cp {} -> {}'.format(
            fast_rcnn_stage2_out['model_path'], final_path)
    shutil.copy(fast_rcnn_stage2_out['model_path'], final_path)
    print 'Final model: {}'.format(final_path)
