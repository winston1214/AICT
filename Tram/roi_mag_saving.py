import csv
tmp=[]
for name in range(2,341): # magnitude 파일 2~340(roi)
    with open('{}.csv'.format(name), 'r') as csvfile: # 열고 maximum값 받아온다
        sex = csv.reader(csvfile,delimiter=',')
        for row in sex:
            tmp.append(row)# 그리고 내용을 저장한다.
            if maximum < len(row):
                maximum = len(row)
    with open('output/{}.csv'.format(name),'w') as f: # 새로운 파일 생성하고 헤더 생성
        writer = csv.DictWriter(f,fieldnames=[i for i in range(maximum)])
        writer.writeheader()
    with open('output/{}.csv'.format(name),'a') as a: # 그 새로운 파일에 붙여넣는다
        w = csv.writer(a,delimiter=',')
        for i in tmp:
            w.writerow(i)
        