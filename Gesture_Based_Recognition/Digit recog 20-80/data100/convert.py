#Extract only data from sensordata file
n=100
n=n*10
open("data.txt","w").close()
#n is number opf samples taken for each gesture
for i in range(n):
    fr=open("dataset.txt","r")
    f=open("data.txt","a")
    data=fr.read()
    data=data.split('\n')
    t=data[i]
    t=eval(t)
    t=eval(t['data'])
    for j in range(int(len(t)/6-4)):
        f.write(str(t[6*j])+","+str(t[6*j+1])+","+str(t[6*j+2])+","+str(t[6*j+3])+","+str(t[6*j+4])+","+str(t[6*j+5])+"\n")
    fr.close()
    f.close()