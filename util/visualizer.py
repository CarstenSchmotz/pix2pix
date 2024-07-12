import os
import sys
import ntpath
import time
import numpy as np
from subprocess import Popen, PIPE
from collections import OrderedDict
from . import util, html

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256, use_wandb=False):
    """Save images to the disk."""
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []
    ims_dict = {}
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        if label == 'real_A':  # Handle combined input separately
            rgb_numpy = im[:, :, :3]
            depth_numpy = im[:, :, 3]
            depth_numpy = np.expand_dims(depth_numpy, axis=2).repeat(3, axis=2)
            im_rgb = rgb_numpy
            im_depth = depth_numpy
            image_name_rgb = '%s_%s_rgb.png' % (name, label)
            image_name_depth = '%s_%s_depth.png' % (name, label)
            save_path_rgb = os.path.join(image_dir, image_name_rgb)
            save_path_depth = os.path.join(image_dir, image_name_depth)
            util.save_image(im_rgb, save_path_rgb, aspect_ratio=aspect_ratio)
            util.save_image(im_depth, save_path_depth, aspect_ratio=aspect_ratio)
            ims.append(image_name_rgb)
            ims.append(image_name_depth)
            txts.append(label + '_rgb')
            txts.append(label + '_depth')
            links.append(image_name_rgb)
            links.append(image_name_depth)
            if use_wandb:
                ims_dict[label + '_rgb'] = wandb.Image(im_rgb)
                ims_dict[label + '_depth'] = wandb.Image(im_depth)
        else:
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(im, save_path, aspect_ratio=aspect_ratio)
            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
            if use_wandb:
                ims_dict[label] = wandb.Image(im)
    webpage.add_images(ims, txts, links, width=width)
    if use_wandb:
        wandb.log(ims_dict)

class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information."""

    def __init__(self, opt):
        """Initialize the Visualizer class"""
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        self.use_wandb = opt.use_wandb
        self.wandb_project_name = opt.wandb_project_name
        self.current_epoch = 0
        self.ncols = opt.display_ncols

        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_wandb:
            self.wandb_run = wandb.init(project=self.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
            self.wandb_run._label(repo='CycleGAN-and-pix2pix')

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file."""
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    if label == 'real_A':  # Handle combined input separately
                        rgb_numpy = image_numpy[:, :, :3]
                        depth_numpy = image_numpy[:, :, 3]
                        depth_numpy = np.expand_dims(depth_numpy, axis=2).repeat(3, axis=2)
                        image_numpy_rgb = rgb_numpy
                        image_numpy_depth = depth_numpy
                        label_html_row += '<td>%s</td>' % (label + '_rgb')
                        label_html_row += '<td>%s</td>' % (label + '_depth')
                        images.append(image_numpy_rgb.transpose([2, 0, 1]))
                        images.append(image_numpy_depth.transpose([2, 0, 1]))
                        idx += 2
                    else:
                        label_html_row += '<td>%s</td>' % label
                        images.append(image_numpy.transpose([2, 0, 1]))
                        idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:     # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)
                        if label == 'real_A':  # Handle combined input separately
                            rgb_numpy = image_numpy[:, :, :3]
                            depth_numpy = image_numpy[:, :, 3]
                            depth_numpy = np.expand_dims(depth_numpy, axis=2).repeat(3, axis=2)
                            image_numpy_rgb = rgb_numpy
                            image_numpy_depth = depth_numpy
                            self.vis.image(image_numpy_rgb.transpose([2, 0, 1]), opts=dict(title=label + '_rgb'),
                                           win=self.display_id + idx)
                            idx += 1
                            self.vis.image(image_numpy_depth.transpose([2, 0, 1]), opts=dict(title=label + '_depth'),
                                           win=self.display_id + idx)
                            idx += 1
                        else:
                            self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                           win=self.display_id + idx)
                            idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                if label == 'real_A':  # Handle combined input separately
                    rgb_numpy = image_numpy[:, :, :3]
                    depth_numpy = image_numpy[:, :, 3]
                    depth_numpy = np.expand_dims(depth_numpy, axis=2).repeat(3, axis=2)
                    image_numpy_rgb = rgb_numpy
                    image_numpy_depth = depth_numpy
                    img_path_rgb = os.path.join(self.img_dir, 'epoch%.3d_%s_rgb.png' % (epoch, label))
                    img_path_depth = os.path.join(self.img_dir, 'epoch%.3d_%s_depth.png' % (epoch, label))
                    util.save_image(image_numpy_rgb, img_path_rgb)
                    util.save_image(image_numpy_depth, img_path_depth)
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch + 1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []
                for label, image_numpy in visuals.items():
                    if label == 'real_A':  # Handle combined input separately
                        img_path_rgb = 'epoch%.3d_%s_rgb.png' % (n, label)
                        img_path_depth = 'epoch%.3d_%s_depth.png' % (n, label)
                        ims.append(img_path_rgb)
                        ims.append(img_path_depth)
                        txts.append(label + '_rgb')
                        txts.append(label + '_depth')
                        links.append(img_path_rgb)
                        links.append(img_path_depth)
                    else:
                        img_path = 'epoch%.3d_%s.png' % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values"""
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk"""
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
