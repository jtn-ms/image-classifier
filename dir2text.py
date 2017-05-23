import os

train_path="/media/frank/Data/Database/ImageNet/DoGCat/train/"#Kaggle/train/"
out_path="./filelist/"
for num, subdir in enumerate(sorted(os.listdir(train_path))):
    sub_path = train_path + subdir
    with open(out_path + subdir + '.txt', 'w') as file:
	#file.writelines([f for f in os.listdir(sub_path)])
	for filename in os.listdir(sub_path):
	     file.write(sub_path + '/' + filename)
	     file.write('\n')

with open(out_path + 'sample.txt', 'w') as file:
    filenames = ['cat','dog']#['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
    for filename in filenames:
	fullname=out_path+ filename + '.txt'
	fid=open(fullname)
	lines=fid.readlines()
	test_num=len(lines)
	rand=random.randint(0,test_num)
	i=0
	for line in lines:
	    word=line.split('\n')
	    fname=word[0]
	    if i==rand:
	    	#file.write(fname)
	    	#file.write('\n')
		break
	    i=i+1
