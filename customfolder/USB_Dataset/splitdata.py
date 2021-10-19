import os

path = '/home/nvidia/wilson/new_darknet/darknet/customfolder/USB_Dataset/all_photos/'
cur_path = os.path.abspath(os.getcwd())
ptlt = os.listdir(path)
scale = 0.1
val = int(len(ptlt)//2 * scale)
print(val)

img_files = []
for i in range(len(ptlt)):
    if ptlt[i].split('.')[1] == "png":
        img_files.append(ptlt[i])
with open("train.txt", "w+") as t:
    with open("val.txt", "w+") as v:
        for i in range(len(img_files)):
            if i <= val :
                    v.writelines([path+img_files[i],'\n'])
            else:
                    t.writelines([path+img_files[i],'\n'])
