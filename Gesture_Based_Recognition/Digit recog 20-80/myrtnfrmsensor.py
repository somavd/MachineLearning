import serial
import codecs

serial_port = 'COM3'
baud_rate = 9600
ser = serial.Serial(serial_port, baud_rate)
flag=False
vec=[0.5,0.5,0.5,0.5,0.5,0.5]
with open("data1.txt", 'w+') as f:
    while True:
        line = ser.readline()
        if not flag:
            flag=True
            line=line.decode('utf-8',errors='ignore').strip()
            line=line.split(',')
            line=[eval(x) for x in line]
            for i in range(6):
                if line[i]<vec[i]:
                    #flag=False
                    break
        if flag:
            fw=open("data.txt","w")
            fw.writelines(str(line)[1:-1]+"\n")
            for i in range(53):
                print("Reached...")
                line = ser.readline()
                line=line.decode('utf-8',errors='ignore').strip()
                line=line.split(',')
                line=[eval(x) for x in line]
                #data.append(line)
                fw.writelines(str(line)[1:-1]+"\n")
            ser.close()
            fw.close()
            exit(0)
            flag=False
            