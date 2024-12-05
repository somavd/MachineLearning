fw=open("testdataConv.txt","w")
fw.close()
for i in range(500):
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