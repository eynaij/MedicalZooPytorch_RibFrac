import matplotlib.pyplot as plt
import numpy as np
import sys, os
import re
import math

def extract_columns(data, xid, yid, stride):
    print('Extract target columns from given data')
    if isinstance(data, list):
        y = data.copy()
        x = np.arange(1, len(data)+1)
    elif data.ndim == 1:
        y = data.copy()
        x = np.arange(1, len(y)+1)
    elif data.ndim != 2:
        raise ValueError
    else:
        if (xid is None) and (yid is None) and (data.shape[-1]==2):
            return data[:,0][::stride], data[:,1][::stride]

        if yid is None:
            #print data.shape
            y = data[:, 0]
        else:
            y = data[:, yid]

        if xid is None:
            x = np.arange(1, len(y)+1)
        else:
            x = data[:, xid]

    return x[::stride], y[::stride]

def moving_average(y, average):
    ynew = []
    vprev = y[0]
    scale = average
    for i, yi in enumerate(y):
        vprev = scale*vprev + (1-scale)*yi
        ynew.append(vprev)
    return np.array(ynew)

def grep_data(logstr, pattern_str):
    """
    grep data using regex
    """
    pattern = re.compile(pattern_str)
    data = pattern.findall(logstr)
    return data

def plot_columns(data_file_list, xid=None, yid=None, stride=1, average=0, xlabel=None, ylabel=None, show_grid=False, ymin=None, ymax=None, title=None, show=''):
    """
    Arguments are:
    - xid       column index for x-axis, starting from 0
    - yid       column index for y-axis, 
    """
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    style = '-'
    marker = '' #'o'
    ymin_, ymax_ = 0., 0.
    for i, data_file in enumerate(data_file_list):
        fid =  open(data_file,'r') 
        log_str = "".join([x.strip() for x in fid.readlines()])
        # train_loss = grep_data(log_str,r'Summary train Epoch +\d+:  Loss:\d+.\d+') 
        train_loss = grep_data(log_str,r'Summary train Epoch +\d+:  Loss:\d+.\d+ 	 DSC:\d+.\d+  	Background : \d+.\d+	fraction : \d+.\d+') 
        #loss = grep_data(log_str,r'Data \d+.\d+ (\d+.\d+)\tLoss \d+.\d+ (\d+.\d+)') 
        #loss = grep_data(log_str, r'Data \d+.\d+ (\d+.\d+)  Loss \d+.\d+ (\d+.\d+)' )
        # val_loss = grep_data(log_str,r'Summary val Epoch +\d+:  Loss:\d+.\d+') 
        val_loss = grep_data(log_str,r'Summary val Epoch +\d+:  Loss:\d+.\d+ 	 DSC:\d+.\d+  	Background : \d+.\d+	fraction : \d+.\d+') 
        # lr = grep_data(log_str, r'\tmAP \d+.\d+')
        # Vmap = grep_data(log_str, r' mAP \d+.\d+\t')
        data_train_loss = [1-float(x.split(':')[-1]) for x in train_loss]
        data_val_loss = [1-float(x.split(':')[-1]) for x in val_loss]
        # data_lr = [float(x.split(' ')[-1]) for x in lr]
        # data_Vmap = [float(x.split(' ')[-1].split('\t')[0]) for x in Vmap]
        #data_lossz_zi = [float(x.split(' ')[-1].split('e-0')[-1]) for x in z_loss]
        #data_lossz = map(lambda x,y: np.sqrt(x* pow(10, -y)) * 60 , data_lossz_di, data_lossz_zi)
        #data_test = map(lambda x,y: math.log(x / pow(10, y),10), data_test_di, data_test_zi)
        data_train_loss = [item for item in data_train_loss]
        data_val_loss = [item for item in data_val_loss]

        # data_lr = [item for item in data_lr]
        #data_lossz = [item if item < 10 else 10  for item in data_lossz]
        
        x_train_loss, y_train_loss = extract_columns(data_train_loss, xid, yid, stride)
        x_val_loss, y_val_loss = extract_columns(data_val_loss, xid, yid, stride)
        # x_lr, y_lr = extract_columns(data_lr, xid, yid, stride)
        # x_Vmap, y_Vmap = extract_columns(data_Vmap, xid, yid, stride)
        # import ipdb;ipdb.set_trace() 
        # x_Vmap *= 29
        # y_lr = [ _/100 for _ in y_lr]
        # y_Vmap = [ _/100 for _ in y_Vmap]
        filename = os.path.splitext(os.path.split(data_file)[1])[0].lstrip('log.')
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(x_train_loss, y_train_loss, colors[i%len(colors)]+style+marker, label='train_loss',linewidth=1)
        ax1.plot(x_val_loss, y_val_loss, colors[(i+1)%len(colors)]+style+marker, label='val_loss',linewidth=1)

#        ax1.plot(x_acc_reg, y_acc_reg, colors[2%len(colors)]+style+'o', markersize = 2, label='accuracy_reg',linewidth=1)
        # ax1.set_yticks(np.arange(0,0.8,0.05)) 
        ax1.set_ylabel('loss')
        
        # ax2 = ax1.twinx()
        # ax2.plot(x_lr, y_lr, colors[1]+style+'o',markersize = 2, label='TrainMAP',linewidth=1)
        # ax2.plot(x_Vmap, y_Vmap, colors[2]+style+'o',markersize = 2, label='ValMAP',linewidth=1)
        #ax2.plot(x_lossz, y_lossz, colors[1]+style+'o',markersize = 2, label='loss_z',linewidth=1)
        #ax2.set_yticks(np.arange(0,0.02,0.001)) 
        #ax2.set_ylabel('lr')
        # ax2.set_yticks(np.arange(0,0.8,0.05)) 
        # ax2.set_ylabel('map')
     
    ax1.legend(loc='best')
    # ax1.legend(loc='lower left')
    # ax2.legend(loc='lower right')
    show_grid = True
    plt.title(filename)
    if show_grid:
        ax1.grid(linestyle='--')
    if not show:
        plt.show()
    else:
        show = filename+'.png'
        plt.savefig(show, dpi=300)

    return True

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Command-line tools for plotting using matplotlib')
    parser.add_argument('-i', dest='input', help='structured data file',
                        default='', type=str, nargs='+')
    parser.add_argument('-xid', dest='xid', help='index for x', 
                        default=None, type=int)
    parser.add_argument('-yid', dest='yid', help='index for y',
                        default=None, type=int)
    parser.add_argument('-s', '--stride', help='stride', default=1, type=int)
    parser.add_argument('-a', '--average', help='Moving average scaling factor for intut data',
                        default=0, type=float)
    parser.add_argument('-xl', '--xlabel', help='label string for x-axis', default=None, type=str)
    parser.add_argument('-yl', '--ylabel', help='label string for y-axis', default=None, type=str)
    parser.add_argument('-g', '--grid', help='show grid', action='store_true')
    parser.add_argument('-ymin', '--ymin', help='minimal value along y-axis', default=None, type=float)
    parser.add_argument('-ymax', '--ymax', help='maximal value along y-axis', default=None, type=float)
    parser.add_argument('-tl', '--title', help='figure title', default=None, type=str)
    parser.add_argument('-sv', '--save_to_file', help='save_name', default='temp.png', type=str)

    args = parser.parse_args()
    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)

    #print args.input
    plot_columns(args.input, args.xid, args.yid, stride=args.stride, average=args.average, xlabel=args.xlabel, ylabel=args.ylabel, show_grid=args.grid, ymin=args.ymin, ymax=args.ymax, title=args.title, show=args.save_to_file)



