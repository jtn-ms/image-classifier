import os
import matplotlib.pyplot as plt  
import numpy as np

global accuracy_path
accuracy_path = './accuracy/'#batch-triplet/'
#global modelname
#modelname = 'triplet-loss'#'softmax'
global epochs
epochs = 30
global interval
interval = 2
#global accuracy_type
#accuracy_type = 'top1'#'top5'
global extension
extension = '.txt'
#global fn_out
#fn_out = accuracy_path + modelname + '-' + accuracy_type + extension
global image_size
image_size = 227
#
def draw_roc_curve(fpr,tpr,title='cosine',save_name='accuracy'):
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic using: '+title)
    plt.legend(loc="lower right")
    plt.show()
    #plt.savefig(accuracy_path + modelname+'.png')

def draw_single(modelname,arr,drawname='loss'):
    _, ax = plt.subplots()  
    #print(loss)
    unit = interval
    if  modelname == 'softmax':
	unit = 132#const
    if  drawname == 'loss':	
	ax.set_ylim([0.0, 2.5])
    else:
        ax.set_ylim([0.0, 1.0])
    ratio = len(arr) / epochs
    ax.plot(unit/ratio * np.arange(len(arr)), arr, 'm-+', label=drawname)

    plt.show()
    plt.savefig(accuracy_path + modelname + '_' + drawname + '.png')
    
def draw_plot(modelname,loss,top1,top5):
    _, ax1 = plt.subplots()  
    ax2 = ax1.twinx()  
    #print(loss)
    # loss -> green
    unit = interval
    if modelname == 'softmax':
	unit = 132#const
    ratio = len(loss) / len(top1)
    ax1.set_ylim([0.0, 2.0])
    ax1.plot(unit / ratio * np.arange(len(loss)), loss, 'g-+', label='loss')
    #ax1.legend(loc=2, ncol=3, shadow=True)  
    #print '2'
    # top1 accuracy -> red
    ax2.set_ylim([0.0, 1.0])
    ax2.plot(unit * np.arange(len(top1)), top1, 'r.-', label='top1')
    #ax2.legend(loc=1, ncol=3, shadow=True)  
    # top5 accuracy -> yellow
    ax2.plot(unit * np.arange(len(top5)), top5, 'yo-', label='top5')
    #ax2.legend(loc=1, ncol=3, shadow=True)  
  
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc=2)
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss')  
    ax2.set_ylabel('accuracy') 
    plt.show()
    plt.savefig(accuracy_path + modelname + '.png')
#read
def readfile(filename):
    fid=open(filename)
    lines=fid.readlines()
    test_num=len(lines)
    fid.close()
    i =0
    array_y = []
    array_x = []
    for line in lines:
        word=line.split('\n')
        value=word[0]
        array_y.append(float(value))
        i+=1
    return array_y
def refine_array(array,refine_size):
    if refine_size > len(array):
    	refine_size = len(array)
        return array
    refined = []
    for i in range(refine_size):
      refined.append(array[i])
    return refined

def plot_all(subdir,modelname):
    loss = readfile(accuracy_path + subdir + '/' + modelname + '-loss' + extension)#,epochs
    top1 = readfile(accuracy_path + subdir + '/' + modelname + '-' + 'top1' + extension)#,epochs
    top5 = readfile(accuracy_path + subdir + '/' + modelname + '-' + 'top5' + extension)#,epochs
    #print(loss)
    #print 'dd'
    #print(top5)
    draw_plot(modelname,loss,top1,top5)
    draw_single(modelname,loss)
    draw_single(modelname,top1,'top1')
    draw_single(modelname,top5,'top5')
   #draw
#sumup
def sumup(modelname,accuracy_type):
    outname = accuracy_path + modelname + '-' + accuracy_type + extension
    with open(outname, 'w') as file:
    	for i in range(1,epochs + 1):
    	    iteration = str(i * interval)
    	    fullname=accuracy_path + modelname + '-' + accuracy_type + '-' +iteration + extension
    	    fid=open(fullname)
    	    lines=fid.readlines()
    	    test_num=len(lines)
    	    for line in lines:
       	        word=line.split('\n')
                value=word[0]
                file.write(value)
                file.write('\n')
                break
if __name__ == '__main__':
#    sumup('triplet-loss','top1')
#    sumup('triplet-loss','top5')
    plot_all('softmax','softmax')
    plot_all('triplet','triplet-loss')
    plot_all('batch-triplet','triplet-loss')

