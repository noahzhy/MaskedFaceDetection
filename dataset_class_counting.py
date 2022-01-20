import os
import glob


counter = {
    0:0,
    1:0,
    2:0,
}

for path in glob.glob(r'D:\FMLD_annotations\train\*.txt'):
    print(path)
    txt = open(path, 'r').readlines()
    txt_idx = [i.split(' ')[0] for i in txt]
    for k in txt_idx:
        counter[int(k)] += 1

print(counter)
quit()
