import os


#print("Hello")
#os.system("echo Hello world?")
os.system("mkdir /tmp/wl2337/newlyLabeled")
os.system("echo make directary?")
os.system("ls /tmp/wl2337")
os.system("echo Anything there?")
#os.system("ls /dataset/unlabeled")



os.system('''
while read p; do
    cp /dataset/unlabeled/$p /tmp/wl2337/newlyLabeled/
done < movedFileNames.txt 
''')

os.system("cp newlyLabeled.pt /tmp/wl2337/")
os.system("ls /tmp/wl2337")


#os.system("ls /dataset")i
