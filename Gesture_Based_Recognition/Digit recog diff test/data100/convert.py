#Extract only data from sensordata file
n=100
fw=open("dataConv0.txt","w")
fw.close()
fw=open("dataConv1.txt","w")
fw.close()
fw=open("dataConv2.txt","w")
fw.close()
fw=open("dataConv3.txt","w")
fw.close()
fw=open("dataConv4.txt","w")
fw.close()
fw=open("dataConv5.txt","w")
fw.close()
fw=open("dataConv6.txt","w")
fw.close()
fw=open("dataConv7.txt","w")
fw.close()
fw=open("dataConv8.txt","w")
fw.close()
fw=open("dataConv9.txt","w")
fw.close()
fw=open("data.txt","w")
fw.close()
fw=open("testdataConv.txt","w")
fw.close()

print(n)
#n is number opf samples taken for each gesture
for i in range(n):
    fr=open("data0.txt","r")
    fw=open("dataConv0.txt","a")
    f=open("data.txt","a")
    data=fr.read()
    data=data.split('\n')
    t=data[i]
    t=eval(t)
    t=eval(t['data'])
    for j in range(int(len(t)/6-4)):
        fw.write(str(t[6*j])+","+str(t[6*j+1])+","+str(t[6*j+2])+","+str(t[6*j+3])+","+str(t[6*j+4])+","+str(t[6*j+5])+"\n")
        f.write(str(t[6*j])+","+str(t[6*j+1])+","+str(t[6*j+2])+","+str(t[6*j+3])+","+str(t[6*j+4])+","+str(t[6*j+5])+"\n")
    fr.close()
    fw.close()
    fr=open("data1.txt","r")
    fw=open("dataConv1.txt","a")
    data=fr.read()
    data=data.split('\n')
    t=data[i]
    t=eval(t)
    t=eval(t['data'])
    for j in range(int(len(t)/6-4)):
        fw.write(str(t[6*j])+","+str(t[6*j+1])+","+str(t[6*j+2])+","+str(t[6*j+3])+","+str(t[6*j+4])+","+str(t[6*j+5])+"\n")
        f.write(str(t[6*j])+","+str(t[6*j+1])+","+str(t[6*j+2])+","+str(t[6*j+3])+","+str(t[6*j+4])+","+str(t[6*j+5])+"\n")
    fr.close()
    fw.close()
    fr=open("data2.txt","r")
    fw=open("dataConv2.txt","a")
    data=fr.read()
    data=data.split('\n')
    t=data[i]
    t=eval(t)
    t=eval(t['data'])
    for j in range(int(len(t)/6-4)):
        fw.write(str(t[6*j])+","+str(t[6*j+1])+","+str(t[6*j+2])+","+str(t[6*j+3])+","+str(t[6*j+4])+","+str(t[6*j+5])+"\n")
        f.write(str(t[6*j])+","+str(t[6*j+1])+","+str(t[6*j+2])+","+str(t[6*j+3])+","+str(t[6*j+4])+","+str(t[6*j+5])+"\n")
    fr.close()
    fw.close()
    fr=open("data3.txt","r")
    fw=open("dataConv3.txt","a")
    data=fr.read()
    data=data.split('\n')
    t=data[i]
    t=eval(t)
    t=eval(t['data'])
    for j in range(int(len(t)/6-4)):
        fw.write(str(t[6*j])+","+str(t[6*j+1])+","+str(t[6*j+2])+","+str(t[6*j+3])+","+str(t[6*j+4])+","+str(t[6*j+5])+"\n")
        f.write(str(t[6*j])+","+str(t[6*j+1])+","+str(t[6*j+2])+","+str(t[6*j+3])+","+str(t[6*j+4])+","+str(t[6*j+5])+"\n")
    fr.close()
    fw.close()
    fr=open("data4.txt","r")
    fw=open("dataConv4.txt","a")
    data=fr.read()
    data=data.split('\n')
    t=data[i]
    t=eval(t)
    t=eval(t['data'])
    for j in range(int(len(t)/6-4)):
        fw.write(str(t[6*j])+","+str(t[6*j+1])+","+str(t[6*j+2])+","+str(t[6*j+3])+","+str(t[6*j+4])+","+str(t[6*j+5])+"\n")
        f.write(str(t[6*j])+","+str(t[6*j+1])+","+str(t[6*j+2])+","+str(t[6*j+3])+","+str(t[6*j+4])+","+str(t[6*j+5])+"\n")
    fr.close()
    fw.close()
    fr=open("data5.txt","r")
    fw=open("dataConv5.txt","a")
    data=fr.read()
    data=data.split('\n')
    t=data[i]
    t=eval(t)
    t=eval(t['data'])
    for j in range(int(len(t)/6-4)):
        fw.write(str(t[6*j])+","+str(t[6*j+1])+","+str(t[6*j+2])+","+str(t[6*j+3])+","+str(t[6*j+4])+","+str(t[6*j+5])+"\n")
        f.write(str(t[6*j])+","+str(t[6*j+1])+","+str(t[6*j+2])+","+str(t[6*j+3])+","+str(t[6*j+4])+","+str(t[6*j+5])+"\n")
    fr.close()
    fw.close()
    fr=open("data6.txt","r")
    fw=open("dataConv6.txt","a")
    data=fr.read()
    data=data.split('\n')
    t=data[i]
    t=eval(t)
    t=eval(t['data'])
    for j in range(int(len(t)/6-4)):
        fw.write(str(t[6*j])+","+str(t[6*j+1])+","+str(t[6*j+2])+","+str(t[6*j+3])+","+str(t[6*j+4])+","+str(t[6*j+5])+"\n")
        f.write(str(t[6*j])+","+str(t[6*j+1])+","+str(t[6*j+2])+","+str(t[6*j+3])+","+str(t[6*j+4])+","+str(t[6*j+5])+"\n")
    fr.close()
    fw.close()
    fr=open("data7.txt","r")
    fw=open("dataConv7.txt","a")
    data=fr.read()
    data=data.split('\n')
    t=data[i]
    t=eval(t)
    t=eval(t['data'])
    for j in range(int(len(t)/6-4)):
        fw.write(str(t[6*j])+","+str(t[6*j+1])+","+str(t[6*j+2])+","+str(t[6*j+3])+","+str(t[6*j+4])+","+str(t[6*j+5])+"\n")
        f.write(str(t[6*j])+","+str(t[6*j+1])+","+str(t[6*j+2])+","+str(t[6*j+3])+","+str(t[6*j+4])+","+str(t[6*j+5])+"\n")
    fr.close()
    fw.close()
    fr=open("data8.txt","r")
    fw=open("dataConv8.txt","a")
    data=fr.read()
    data=data.split('\n')
    t=data[i]
    t=eval(t)
    t=eval(t['data'])
    for j in range(int(len(t)/6-4)):
        fw.write(str(t[6*j])+","+str(t[6*j+1])+","+str(t[6*j+2])+","+str(t[6*j+3])+","+str(t[6*j+4])+","+str(t[6*j+5])+"\n")
        f.write(str(t[6*j])+","+str(t[6*j+1])+","+str(t[6*j+2])+","+str(t[6*j+3])+","+str(t[6*j+4])+","+str(t[6*j+5])+"\n")
    fr.close()
    fw.close()
    fr=open("data9.txt","r")
    fw=open("dataConv9.txt","a")
    data=fr.read()
    data=data.split('\n')
    t=data[i]
    t=eval(t)
    t=eval(t['data'])
    for j in range(int(len(t)/6-4)):
        fw.write(str(t[6*j])+","+str(t[6*j+1])+","+str(t[6*j+2])+","+str(t[6*j+3])+","+str(t[6*j+4])+","+str(t[6*j+5])+"\n")
        f.write(str(t[6*j])+","+str(t[6*j+1])+","+str(t[6*j+2])+","+str(t[6*j+3])+","+str(t[6*j+4])+","+str(t[6*j+5])+"\n")
    fr.close()
    fw.close()
    f.close()
for i in range(110):
    fr=open("datatest.txt","r")
    fw=open("testdataConv.txt","a")
    data=fr.read()
    data=data.split('\n')
    t=data[i]
    t=eval(t)
    t=eval(t['data'])
    for j in range(int(len(t)/6-4)):
        fw.write(str(t[6*j])+","+str(t[6*j+1])+","+str(t[6*j+2])+","+str(t[6*j+3])+","+str(t[6*j+4])+","+str(t[6*j+5])+"\n")
    fr.close()
    fw.close()
fr=open("data.txt","r")
fw=open("dataall.txt","w")
data=fr.read()
for i in range(10):
    fw.write(data)
fr.close()
fw.close()