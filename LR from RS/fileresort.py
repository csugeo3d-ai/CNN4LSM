import os

# Renumbering files and folders in the same parent directory
def rename(path):
    num = 1
    for filepath, dirnames, filenames in os.walk(path):
        filenames.sort(key=lambda x: int(x.split('.')[0]))
        for filename in filenames:
            used_name = os.path.join(filepath, filename)
            print('used_name = ' +'a'+ used_name)
            extension = os.path.splitext(used_name)[-1]
            new_name = os.path.join(filepath, 'a'+ str(num) + extension)
            os.rename(used_name, new_name)
            print('new_name=' + new_name)
            num += 1

if __name__ == "__main__":
    rename("E:/huan/tiffdatawulingyuan/smalldata/secondprc/")