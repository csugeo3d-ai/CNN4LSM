import os

# Enter the folder address
path = "E:/huan/tiffdatawulingyuan/smalldata/secondprc/label1/"
files = os.listdir(path)

# Output all filenames, just to see
for file in files:
    print(file)

# Getting the old name and the new name
i = 0
for file in files:
    # old Information on old names
    old = path + os.sep + files[i]
    # new is the information of the new name, the operation here is to delete the top 'Koala is busy o - ' total 8 characters
    new = path + os.sep + file.replace('a','')
    #new = "E:/huan/tiffdatawulingyuan/smalldata/" + os.sep + file[8:]
    # replace the old with the new
    os.rename(old,new)
    i+=1