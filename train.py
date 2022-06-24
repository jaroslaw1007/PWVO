import tensorflow as tf
import numpy as np

import sys
import time
import argparse
import os

from core.net.VONet import VONet
from core.net.PWVO import PWVO

from core.dataset.sintel_dataset import SintelDataLoader
from core.dataset.custom_dataset import CustomDataLoader

from core.utils.viz_utils import visualize_unc, visualize
from core.utils.utils import create_directory, save_func

from trainer import Trainer
from evaluate import evaluate_PWVO, evaluate_VONet

loss_name_list = ['total_loss', 'rotation_loss', 'translation_loss', 'flow_loss', 'depth_loss']
metrics_name_list = ['r_metrics', 't_metrics', 'flow_metrics', 'depth_metrics']

parser = argparse.ArgumentParser(description='VisualOdometry')
    
parser.add_argument('--height', type=int, default=128, help='Height of input')
parser.add_argument('--width', type=int, default=256, help='Width of input')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--dataset_size', type=int, default=100000, help='Total samples of dataset')
parser.add_argument('--training_sample', type=int, default=100000, help='Training samples of dataset')

parser.add_argument('--img_freq', type=int, default=1000, help='Frequency of recording visualization')
parser.add_argument('--log_freq', type=int, default=500, help='Frequency of logging tensorboard')
parser.add_argument('--save_freq', type=int, default=10, help='Frequency of saving model weights')

parser.add_argument('--data_path', type=str, required=True, help='Directory path of your input training datas')
parser.add_argument('--detail', type=str, required=True, help='Save directory name')
parser.add_argument('--img_path', type=str, default='visual', help='Directory path of saving visualization')
parser.add_argument('--ckpt_path', type=str, default='ckpt', help='Directory path of saving model weights')
parser.add_argument('--log_path', type=str, default='log', help='Directory path of saving tensorboard log')
parser.add_argument('--txt_path', type=str, default='txt', help='Directory path of saving experiment txt file')

parser.add_argument('--eval_data_path', type=str, required=True, help='Directory path of your input evaluation datas')
parser.add_argument('--eval_img_path', type=str, default='visual_sintel', \
                    help='Directory path of saving evaluation visualization')

parser.add_argument('--baseline', action='store_true', default=False, help='Model VONet or PWVO')
parser.add_argument('--coord', action='store_true', default=False, help='Add coordinate layer or not')
parser.add_argument('--ego', action='store_true', default=False, help='Input ego or total')

parser.add_argument('--selection', type=str, choices=['mean_select', 'top_k_select', \
                                                      'patch_select', 'patch_soft_select', \
                                                      'soft_select'], help='Different types of Selection Module')

parser.add_argument('--no_extrinsic', action='store_true', default=False, help='Rotation and translation loss')
parser.add_argument('--ego_motion', action='store_true', default=False, help='Ego flow loss')
parser.add_argument('--depth', action='store_true', default=False, help='Depth loss')

parser.add_argument('--rt_unc', action='store_true', default=False, help='Using uncertainty technique on rotation and translation loss')
parser.add_argument('--f_unc', action='store_true', default=False, help='Using uncertainty technique on ego flow loss')
parser.add_argument('--d_unc', action='store_true', default=False, help='Using uncertainty technique on depth loss')

parser.add_argument('--data_mean', type=float, default=0.0)
parser.add_argument('--data_std', type=float, default=200.0)
parser.add_argument('--flow_delta', type=float, default=20.0)
parser.add_argument('--patch_size', type=int, default=32)
parser.add_argument('--top_k', type=int, default=5)

parser.add_argument('--r_weight', type=float, default=1.0)
parser.add_argument('--t_weight', type=float, default=1.0)
parser.add_argument('--f_weight', type=float, default=1.0)
parser.add_argument('--d_weight', type=float, default=1.0)
parser.add_argument('--func_scale', type=float, default=0.1)
parser.add_argument('--dunc_scale', type=float, default=0.25)

args = parser.parse_args()

class Logger:
    def __init__(self, logdir):
        self.logdir = logdir
        self.writer = tf.summary.create_file_writer(self.logdir)
        
    def _log_value(self, metrics, steps):
        with self.writer.as_default():
            for k in metrics:
                tf.summary.scalar(k, metrics[k], steps)

class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, lr_decay_step=25000, lr_decay_factor=0.5):
        
        self.initial_lr = initial_lr
        self.lr_decay_step = lr_decay_step
        self.lr_decay_factor = lr_decay_factor

    def __call__(self, step):        
        return self.initial_lr * (self.lr_decay_factor ** (step // self.lr_decay_step))

def train(dataset, eval_dataset, net_trainer, optimizer):
    # Initialize Model
    if args.baseline:
        net = VONet(args)
    else:
        net = PWVO(args)
    
    # Create directories
    log_dir = os.path.join(args.log_path, args.detail)
    img_dir = os.path.join(args.img_path, args.detail)
    ckpt_dir = os.path.join(args.ckpt_path, args.detail)
    
    create_directory(log_dir)
    create_directory(img_dir)
    create_directory(ckpt_dir)
    
    # Create tensorboard logger
    logger = Logger(log_dir)
    
    loss_list = [tf.keras.metrics.Mean(name=loss_name) for loss_name in loss_name_list]
    metrics_list = [tf.keras.metrics.Mean(name=metrics_name) for metrics_name in metrics_name_list]
    
    global_step = 0

    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        for l in loss_list: l.reset_state()
        for m in metrics_list: m.reset_state()

        for batch_idx, inputs in enumerate(dataset):
            global_step += 1
            
            if args.baseline:
                train_output = net_trainer.train_VONet_step(net, optimizer, inputs)
            else:
                train_output = net_trainer.train_PWVO_step(net, optimizer, inputs)
            
            for idx, loss_name in enumerate(loss_name_list): loss_list[idx](train_output[loss_name])
            for idx, metrics_name in enumerate(metrics_name_list): metrics_list[idx](train_output[metrics_name])
        
            print(('Epoch {} Batch {} Global Step {} Loss {:.3f} R Loss {:.3f} T Loss {:.3f} ' + \
               'F Loss {:.3f} D Loss {:.3f}').format(epoch + 1, batch_idx + 1, \
                global_step, loss_list[0].result(), loss_list[1].result(), loss_list[2].result(), \
                loss_list[3].result(), loss_list[4].result()), end='\r')
            
            if global_step % args.log_freq == 0:
                metrics_dict = {}
                
                for idx, loss_name in enumerate(loss_name_list): metrics_dict[loss_name] = loss_list[idx].result()
                for idx, metrics_name in enumerate(metrics_name_list): metrics_dict[metrics_name] = metrics_list[idx].result()
                
                logger._log_value(metrics_dict, global_step)
            
            if global_step % args.img_freq == 0:
                if args.baseline:
                    visualize(inputs[0][:4], inputs[1][:4], train_output['ego_motion_recon'][:4], \
                              train_output['flow_error_map'][:4], img_dir, global_step)
                else:
                    unc_rx, unc_ry, unc_rz, unc_tx, unc_ty, unc_tz = tf.split(train_output['uncertainty_map'], \
                                                                          num_or_size_splits=6, axis=3)
                    
                    visualize_unc(inputs[0][:4], inputs[1][:4], train_output['ego_motion_recon'][:4], \
                                  unc_rx[:4], train_output['flow_error_map'][:4], img_dir, global_step)

        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            net.save_weights(os.path.join(ckpt_dir, 'Epoch_' + str(epoch + 1)))
            
            if args.baseline:
                evaluate_VONet(args, eval_dataset, net, epoch + 1)
            else:
                evaluate_PWVO(args, eval_dataset, net, epoch + 1)

        print('\n')
        print('Epoch Times {}'.format(time.time() - epoch_start))
        print('\n')
        
if __name__ == '__main__':
    train_dataloader = CustomDataLoader(args)
    eval_dataloader = SintelDataLoader(args)
    
    train_dataset = train_dataloader.generate_train_dataset()
    eval_dataset = eval_dataloader.generate_dataset()
    optimizer = tf.keras.optimizers.Adam(learning_rate=MyLRSchedule(args.lr), beta_1=0.5, beta_2=0.999)
    net_trainer = Trainer(args)
        
    train(train_dataset, eval_dataset, net_trainer, optimizer)