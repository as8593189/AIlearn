#导包
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

#读取文件
all_mail=[]
all_labels=[]
test_mail=[]
test_labels=[]
count=0
f=open("sms_spam.txt","rb")
while True:
    line=f.readline().decode("utf-8")
    if count==0:
        count=count+1
        continue
    if line:
        count=count+1
        line=line.split(",")
        label=line[0]
        mail=line[1]
        all_mail.append(mail)
        if label=="ham":
            all_labels.append(1)
        elif label=="spam":
            all_labels.append(0)
        if count>=5550:
            test_mail.append(mail)
            if label=="ham":
                test_labels.append(1)
            else:
                test_labels.append(0)
    else:
        break

#特征向量转换
cv=CountVectorizer()
cv_fit=cv.fit_transform(all_mail)
cv_test=CountVectorizer(vocabulary=cv.vocabulary_)
cv_test_fit=cv_test.fit_transform(test_mail)
nb=MultinomialNB(alpha=1)
nb.fit(cv_fit,all_labels)
pred=nb.predict(cv_test_fit)
for i,v in enumerate(pred):
    if v==1:
        print("正常邮件",end="\t")

    else:
        print("垃圾邮件", end="\t")


