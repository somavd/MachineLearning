import random
n=10
fw=open("datashuffle.txt","w")
fr=open("data0.txt","r")
data=fr.read()
fw.write(data+"\n")
fr.close()
fr=open("data1.txt","r")
data=fr.read()
fw.write(data+"\n")
fr.close()
fr=open("data2.txt","r")
data=fr.read()
fw.write(data+"\n")
fr.close()
fr=open("data3.txt","r")
data=fr.read()
fw.write(data+"\n")
fr.close()
fr=open("data4.txt","r")
data=fr.read()
fw.write(data+"\n")
fr.close()
fr=open("data5.txt","r")
data=fr.read()
fw.write(data+"\n")
fr.close()
fr=open("data6.txt","r")
data=fr.read()
fw.write(data+"\n")
fr.close()
fr=open("data7.txt","r")
data=fr.read()
fw.write(data+"\n")
fr.close()
fr=open("data8.txt","r")
data=fr.read()
fw.write(data+"\n")
fr.close()
fr=open("data9.txt","r")
data=fr.read()
fw.write(data)
fr.close()
fw.close()

open("dataset.txt","w").close()
open("output.txt","w").close()
fr=open("datashuffle.txt","r")
data=fr.read()
data=list(eval(data.replace('\n',',')))
fr.close()
op=[]
for i in range(10):
    for j in range(n):
        op.append(i)
fwd=open("dataset.txt","a+")
fwo=open("output.txt","a+")
t=n*10-1;
for i in range(t,-1,-1):
    k=random.randint(0,i);
    fwd.write(str(data[k])+"\n")
    fwo.write(str(op[k])+"\n")
    del data[k]
    del op[k]
fwd.close()
fwo.close()
'''fwd=open("dataset.txt","r+")
fwo=open("output.txt","r+")
data=fwd.read()
print(data)
fwd.write(data)
fwd.write(data)
fwd.write(data)
fwd.write(data)
data=fwo.read()
fwo.write(data)
fwo.write(data)
fwo.write(data)
fwo.write(data)
fwd.close()
fwo.close()'''