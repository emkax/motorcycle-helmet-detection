import csv

with open("./yolov8l_model/results.csv",newline='') as f:
    readed = csv.reader(f,delimiter=' ', quotechar='|')
    data = [value[0].split(",") for i,value in enumerate(readed)]

print(data[0])
time = 0 #1
avg_train_loss = 0 #2 3 4
avg_val_loss = 0 #9 10 11
avg_precision = 0 #5
avg_recall = 0 #6
avg_mAP50 = 0 #7
avg_mAP5095 = 0 #8

for i in range(1,len(data)):
    avg_train_loss += float(data[i][2]) + float(data[i][3]) + float(data[i][4])
    avg_val_loss += float(data[i][9]) + float(data[i][10]) + float(data[i][11])
    avg_precision += float(data[i][5])
    avg_recall += float(data[i][6])
    avg_mAP50 += float(data[i][7])
    avg_mAP5095 +=float(data[i][8])

print(avg_train_loss/(len(data)-1))
print(avg_val_loss/(len(data)-1))
print(avg_precision/(len(data)-1))
print(avg_recall/(len(data)-1))
print(avg_mAP50/(len(data)-1))
print(avg_mAP5095/(len(data)-1))

