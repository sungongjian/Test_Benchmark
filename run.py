from caffe2.python import core,workspace
from caffe2.proto import caffe2_pb2
from caffe2.python.models import bvlc_alexnet as mynet
import numpy as np
from numpy import append
from matplotlib.pyplot import imread
import os
import argparse
import time

def cut(pic, size):
	"this reduce pic from [1, n, n, 3] to [1, size, size, 3] by keeping its center"
	hw = pic[0].shape[0]
	head = (hw-size)//2
	tail = hw-size-head
	new = np.delete(pic[0], slice(hw-tail, hw),axis=0)
	new = np.delete(new, slice(0,head), axis=0)
	new = np.delete(new, slice(hw-tail, hw),axis=1)
	new = np.delete(new, slice(0,head), axis=1)
	return [new]
	

def readin(root_dir, size):
	"This read all files that evaluates true by judge(name) to data"
	global data
	global data_list
	for (dirpath, dirnames, filenames) in os.walk(root_dir):
		# first deal with files
		for filename in filenames:
			if filename.endswith('.jpg'):
				new = imread(os.path.join(dirpath, filename))
				# here new is (256, 256, 3) or (256, 256, 4)
				if(new.ndim!=3): continue
				if(new.shape[2]!=3 and new.shape[2]!=4): continue
				new = cut([new], size)[0]
				new = new.swapaxes(0,2)
				new = new.swapaxes(1,2)
				if(new.shape[0]==4): new = np.delete(new, (3), axis=0)
				if(data_list[0]==data_list[1]):
					data.resize(data_list[1]+100,3,args.s,args.s)
					data_list[1] += 100
				np.copyto(data[data_list[0]],new)
				data_list[0] += 1
				if(data_list[0]%100==0): 
					elapsed = time.time()-start_time
					print('num of read in files: '+str(data_list[0])+' @ time '+str(elapsed))
				if(data_list[0]>=args.n): break
		if(data_list[0]>=args.n): break

parser = argparse.ArgumentParser(description = 'specify work directory and size')
parser.add_argument('-d', metavar = 'D', type = str, default = '.', help = 'set working root')
parser.add_argument('-s', metavar = 'S', type = int, default = 227, help = 'set height and width in computing')
parser.add_argument('-n', metavar = 'N', type = int, default = 400, help = 'set batch size(number of images)')
# parser.print_help()
args = parser.parse_args()
			
init_net = mynet.init_net
predict_net = mynet.predict_net
predict_net.name = "alexnet_test"

# run on GPU, device id: 1, you can change to 0
device_opts = core.DeviceOption(caffe2_pb2.CUDA, 1)
init_net.device_option.CopyFrom(device_opts)
predict_net.device_option.CopyFrom(device_opts)

# use files as input, format of data: NCHW
# if you want to use random inputs, comment these codes
# until the next comment and insert your code to init data.
# data has to be this shape (any_batch_size, 3, args.s, args.s)
data = np.zeros(shape=(100,3,args.s,args.s), dtype=np.float32)
data_list = [0, 100]
start_time = time.time()
readin(args.d, args.s)
data.resize(data_list[0],3,args.s,args.s)
# for test reason, we want 32 to divide batch_size
multi = data.shape[0]//32
data = np.delete(data, slice(32*multi, data.shape[0]), axis=0)
print(data.shape[0])
if(data.size==0): exit()

# now start run
core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
workspace.FeedBlob('data', data, device_opts)
workspace.RunNetOnce(init_net)
workspace.CreateNet(predict_net)
# workspace.RunNet(predict_net.name, 1)
workspace.TestBenchmark(predict_net.name, 0, 1, True)
res = workspace.FetchBlob(predict_net.external_output[-1])
print(str(np.argmax(res, axis=1)))
