import os

# the last '/' slash don't forget!!! or it will still training but garbage weight
path = '/workspace/wilson/dataset/singal_usb_blackBG_211102/all_211102/'
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
