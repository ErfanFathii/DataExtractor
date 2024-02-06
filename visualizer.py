import numpy as np
from scipy import stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



data = np.genfromtxt(r'C:\Users\AC TI VE\Desktop\gaussian\Log10.csv' , delimiter=',')

actual_dist_player = []
gaussian_player = np.array([])
quantized_player = []
listgen = [[]]


for row in data:
    if row[1] == 2:
        #np.append(pusher[0],row[0])
        pusher = []
        pusher.append(row[0])
        pusher.append(row[2])
        pusher.append(row[3])
        #np.append(pusher[1],row[2])
        #np.append(pusher[2],row[3])
        listgen.append([pusher])
        #print(listgen)
        #listgen.append([row[0],row[2],row[3]])
        actual_dist_player.append(row[0])
        #gaussian_player.append(row[2])
        quantized_player.append(row[3])
# Create a dataframe
#print("valfirst = ",listgen)
devidedlist1 = [[]]
devidedlist2 = [[]]
devidedlist3 = [[]]
devidedlist4 = [[]]
devidedlist5 = [[]]
devidedlist6 = [[]]
devidedlist7 = [[]]
devidedlist8 = [[]]
devidedlist9 = [[]]
devidedlist10 = [[]]
devidedlist11 = [[]]
devidedlist12 = [[]]
devidedlist13 = [[]]
devidedlist14 = [[]]
devidedlist15 = [[]]
devidedlist16 = [[]]
devidedlist17 = [[]]
devidedlist18 = [[]]
devidedlist19 = [[]]
devidedlist20 = [[]]
devidedlist21 = [[]]
devidedlist22 = [[]]
devidedlist23 = [[]]
devidedlist24 = [[]]
devidedlist25 = [[]]
devidedlist26 = [[]]
devidedlist27 = [[]]
devidedlist28 = [[]]
devidedlist29 = [[]]
devidedlist30 = [[]]
devidedlist31 = [[]]
devidedlist32 = [[]]
devidedlist33 = [[]]
devidedlist34 = [[]]




for i in listgen : 
    for j in i :
        if j[2]  >= 0.6 and j[2] <= 0.7 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist1.append([pusher2]) 
        if j[2]  >= 0.7 and j[2] <= 0.8 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist2.append([pusher2])

        if j[2]  >= 0.8 and j[2] <= 0.9 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist3.append([pusher2])

        if j[2]  >= 0.9 and j[2] <= 1 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist4.append([pusher2])
            
        if j[2]  >= 1 and j[2] <= 1.1 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist5.append([pusher2])

        if j[2]  >= 1.1 and j[2] <= 1.2 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist6.append([pusher2])

        if j[2]  >= 1.2 and j[2] <= 1.3 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist7.append([pusher2])

        if j[2]  >= 1.3 and j[2] <= 1.5 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist8.append([pusher2])

        if j[2]  >= 1.5 and j[2] <= 1.6 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist9.append([pusher2])

        if j[2]  >= 1.6 and j[2] <= 1.8 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist10.append([pusher2])

        if j[2]  >= 1.8 and j[2] <= 10 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist11.append([pusher2])

        ##
        if j[2]  >= 10 and j[2] <= 11 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist12.append([pusher2]) 
        if j[2]  >= 11 and j[2] <= 12.2 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist13.append([pusher2])

        if j[2]  >= 12.2 and j[2] <= 13.5 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist14.append([pusher2])

        if j[2]  >= 13.5 and j[2] <= 14.9 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist15.append([pusher2])
            
        if j[2]  >= 14.9 and j[2] <= 16.4 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist16.append([pusher2])

        if j[2]  >= 16.4 and j[2] <= 18.2 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist17.append([pusher2])

        if j[2]  >= 18.2 and j[2] <= 20.1 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist18.append([pusher2])

        if j[2]  >= 20.1 and j[2] <= 22.2 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist19.append([pusher2])

        if j[2]  >= 22.2 and j[2] <= 24.5 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist20.append([pusher2])

        if j[2]  >= 24.5 and j[2] <= 27.1 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist21.append([pusher2])

        if j[2]  >= 27.1 and j[2] <= 30 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist22.append([pusher2])

        ##2
        if j[2]  >= 30 and j[2] <= 33.1 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist23.append([pusher2]) 
        if j[2]  >= 33.1 and j[2] <= 36.6 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist24.append([pusher2])

        if j[2]  >= 36.6 and j[2] <= 40.4 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist25.append([pusher2])

        if j[2]  >= 40.4 and j[2] <= 44.7 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist26.append([pusher2])
            
        if j[2]  >= 44.7 and j[2] <= 49.4 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist27.append([pusher2])

        if j[2]  >= 49.4 and j[2] <= 54.6 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist28.append([pusher2])

        if j[2]  >= 54.6 and j[2] <= 60.3 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist29.append([pusher2])

        if j[2]  >= 60.3 and j[2] <= 66.7 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist30.append([pusher2])

        if j[2]  >= 66.7 and j[2] <= 73.7 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist31.append([pusher2])

        if j[2]  >= 73.7 and j[2] <= 81.5 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist32.append([pusher2])

        if j[2]  >= 81.5 and j[2] <= 90 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist33.append([pusher2])

        if j[2]  >= 90 and j[2] <= 99.5 :
            pusher2 = []
            pusher2.append(j[0])
            pusher2.append(j[1])
            pusher2.append(j[2])
            devidedlist34.append([pusher2])




firstact = []
for i in devidedlist1 :
    for j in i :
        firstact.append(j[0])
firstnoise = []
for i in devidedlist1 :
    for j in i :
        firstnoise.append(j[1])


secact = []
for i in devidedlist2 :
    for j in i :
        secact.append(j[0])
secnoise = []
for i in devidedlist2 :
    for j in i :
        secnoise.append(j[1])

thirdact = []
for i in devidedlist3 :
    for j in i :
        thirdact.append(j[0])
threenoise = []
for i in devidedlist3 :
    for j in i :
        threenoise.append(j[1])

foact = []
for i in devidedlist4 :
    for j in i :
        foact.append(j[0])
fonoise = []
for i in devidedlist4 :
    for j in i :
        fonoise.append(j[1])

fiact = []
for i in devidedlist5 :
    for j in i :
        fiact.append(j[0])
finoise = []
for i in devidedlist5 :
    for j in i :
        finoise.append(j[1])


sixact = []
for i in devidedlist6 :
    for j in i :
        sixact.append(j[0])
sixnoise = []
for i in devidedlist6 :
    for j in i :
        sixnoise.append(j[1])

sevact = []
for i in devidedlist7 :
    for j in i :
        sevact.append(j[0])
sevnoise = []
for i in devidedlist7 :
    for j in i :
        sevnoise.append(j[1])

eiact = []
for i in devidedlist8 :
    for j in i :
        eiact.append(j[0])
einoise = []
for i in devidedlist8 :
    for j in i :
        einoise.append(j[1])

niact = []
for i in devidedlist9 :
    for j in i :
        niact.append(j[0])
ninoise = []
for i in devidedlist9 :
    for j in i :
        ninoise.append(j[1])

teact = []
for i in devidedlist10 :
    for j in i :
        teact.append(j[0])
tenoise = []
for i in devidedlist10 :
    for j in i :
        tenoise.append(j[1])

eeact = []
for i in devidedlist11 :
    for j in i :
        eeact.append(j[0])
eenoise = []
for i in devidedlist11 :
    for j in i :
        eenoise.append(j[1])


##
teact = []
for i in devidedlist12 :
    for j in i :
        teact.append(j[0])
tenoise = []
for i in devidedlist12 :
    for j in i :
        tenoise.append(j[1])

thteenact = []
for i in devidedlist13 :
    for j in i :
        thteenact.append(j[0])
thteennoise = []
for i in devidedlist13 :
    for j in i :
        thteennoise.append(j[1])

ftact = []
for i in devidedlist14 :
    for j in i :
        ftact.append(j[0])
ftnoise = []
for i in devidedlist14 :
    for j in i :
        ftnoise.append(j[1])

fitact = []
for i in devidedlist15 :
    for j in i :
        fitact.append(j[0])
fitnoise = []
for i in devidedlist15 :
    for j in i :
        fitnoise.append(j[1])


sixtact = []
for i in devidedlist16 :
    for j in i :
        sixtact.append(j[0])
sixtnoise = []
for i in devidedlist16 :
    for j in i :
        sixtnoise.append(j[1])

sevtact = []
for i in devidedlist17 :
    for j in i :
        sevtact.append(j[0])
sevtnoise = []
for i in devidedlist17 :
    for j in i :
        sevtnoise.append(j[1])

eitact = []
for i in devidedlist18 :
    for j in i :
        eitact.append(j[0])
eitnoise = []
for i in devidedlist18 :
    for j in i :
        eitnoise.append(j[1])

nitact = []
for i in devidedlist19 :
    for j in i :
        nitact.append(j[0])
nitnoise = []
for i in devidedlist19 :
    for j in i :
        nitnoise.append(j[1])

tact = []
for i in devidedlist20 :
    for j in i :
        tact.append(j[0])
tnoise = []
for i in devidedlist20 :
    for j in i :
        tnoise.append(j[1])

toact = []
for i in devidedlist21 :
    for j in i :
        toact.append(j[0])
tonoise = []
for i in devidedlist21 :
    for j in i :
        tonoise.append(j[1])

##2
ttact = []
for i in devidedlist22 :
    for j in i :
        ttact.append(j[0])
ttnoise = []
for i in devidedlist22 :
    for j in i :
        ttnoise.append(j[1])

tthact = []
for i in devidedlist23 :
    for j in i :
        tthact.append(j[0])
tthnoise = []
for i in devidedlist23 :
    for j in i :
        tthnoise.append(j[1])

tfact = []
for i in devidedlist24 :
    for j in i :
        tfact.append(j[0])
tfnoise = []
for i in devidedlist24 :
    for j in i :
        tfnoise.append(j[1])

tfiact = []
for i in devidedlist25 :
    for j in i :
        tfiact.append(j[0])
tfinoise = []
for i in devidedlist25 :
    for j in i :
        tfinoise.append(j[1])

tsact = []
for i in devidedlist26 :
    for j in i :
        tsact.append(j[0])
tsnoise = []
for i in devidedlist26 :
    for j in i :
        tsnoise.append(j[1])


tseact = []
for i in devidedlist27 :
    for j in i :
        tseact.append(j[0])
tsenoise = []
for i in devidedlist27 :
    for j in i :
        tsenoise.append(j[1])

teact = []
for i in devidedlist28 :
    for j in i :
        teact.append(j[0])
tenoise = []
for i in devidedlist28 :
    for j in i :
        tenoise.append(j[1])

tniact = []
for i in devidedlist29 :
    for j in i :
        tniact.append(j[0])
tninoise = []
for i in devidedlist29 :
    for j in i :
        tninoise.append(j[1])

thact = []
for i in devidedlist30 :
    for j in i :
        thact.append(j[0])
thnoise = []
for i in devidedlist30 :
    for j in i :
        thnoise.append(j[1])

thoact = []
for i in devidedlist31 :
    for j in i :
        thoact.append(j[0])
thonoise = []
for i in devidedlist31 :
    for j in i :
        thonoise.append(j[1])

thtact = []
for i in devidedlist32 :
    for j in i :
        thtact.append(j[0])
thtnoise = []
for i in devidedlist32 :
    for j in i :
        thtnoise.append(j[1])

ththact = []
for i in devidedlist33 :
    for j in i :
        ththact.append(j[0])
ththnoise = []
for i in devidedlist33 :
    for j in i :
        ththnoise.append(j[1])


thfact = []
for i in devidedlist34 :
    for j in i :
        thfact.append(j[0])
thfnoise = []
for i in devidedlist34 :
    for j in i :
        thfnoise.append(j[1])






datap = {'x': firstact, 'y': firstnoise}
df = pd.DataFrame(datap)
datap2 = {'x': secact, 'y': secnoise}
df2 = pd.DataFrame(datap2)
datap3 = {'x': thirdact, 'y': threenoise}
df3 = pd.DataFrame(datap3)
datap4 = {'x': foact, 'y': fonoise}
df4 = pd.DataFrame(datap4)
datap5 = {'x': fiact, 'y': finoise}
df5 = pd.DataFrame(datap5)
datap6 = {'x': sixact, 'y': sixnoise}
df6 = pd.DataFrame(datap6)
datap7 = {'x': sevact, 'y': sevnoise}
df7 = pd.DataFrame(datap7)
datap8 = {'x': eiact, 'y': einoise}
df8 = pd.DataFrame(datap8)
datap9 = {'x': niact, 'y': ninoise}
df9 = pd.DataFrame(datap9)
datap10 = {'x': teact, 'y': tenoise}
df10 = pd.DataFrame(datap10)
datap11 = {'x': eeact, 'y': eenoise}
df11 = pd.DataFrame(datap11)

#
datap12 = {'x': teact, 'y': tenoise}
df12 = pd.DataFrame(datap12)
datap13 = {'x': thteenact, 'y': thteennoise}
df13 = pd.DataFrame(datap13)
datap14 = {'x': ftact, 'y': ftnoise}
df14 = pd.DataFrame(datap14)
datap15 = {'x': fitact, 'y': fitnoise}
df15 = pd.DataFrame(datap15)
datap16 = {'x': sixtact, 'y': sixtnoise}
df16 = pd.DataFrame(datap16)
datap17 = {'x': sevtact, 'y': sevtnoise}
df17 = pd.DataFrame(datap17)
datap18 = {'x': eitact, 'y': eitnoise}
df18 = pd.DataFrame(datap18)
datap19 = {'x': nitact, 'y': nitnoise}
df19 = pd.DataFrame(datap19)
datap20 = {'x': tact, 'y': tnoise}
df20 = pd.DataFrame(datap20)
datap21 = {'x': toact, 'y': tonoise}
df21 = pd.DataFrame(datap21)
datap22 = {'x': ttact, 'y': ttnoise}
df22 = pd.DataFrame(datap22)
#
datap23 = {'x': tthact, 'y': tthnoise}
df23 = pd.DataFrame(datap23)
datap24 = {'x': tfact, 'y': tfnoise}
df24 = pd.DataFrame(datap24)
datap25 = {'x': tfiact, 'y': tfinoise}
df25 = pd.DataFrame(datap25)
datap26 = {'x': tsact, 'y': tsnoise}
df26 = pd.DataFrame(datap26)
datap27 = {'x': tseact, 'y': tsenoise}
df27 = pd.DataFrame(datap27)
datap28 = {'x': teact, 'y': tenoise}
df28 = pd.DataFrame(datap28)
datap29 = {'x': tniact, 'y': tninoise}
df29 = pd.DataFrame(datap29)
datap30 = {'x': thact, 'y': thnoise}
df30 = pd.DataFrame(datap30)
datap31 = {'x': thoact, 'y': thonoise}
df31 = pd.DataFrame(datap31)
datap32 = {'x': thtact, 'y': thtnoise}
df32 = pd.DataFrame(datap32)
datap33 = {'x': ththact, 'y': ththnoise}
df33 = pd.DataFrame(datap33)
datap34 = {'x': thfact, 'y': thfnoise}
df34 = pd.DataFrame(datap34)

ax = plt.axes()
ax.set_facecolor("black")

# Create a 2D density plot
sns.kdeplot(data=df, x="x", y="y", fill=True,  cmap="copper",)
sns.kdeplot(data=df2, x="x", y="y", fill=True,  cmap="copper",)
sns.kdeplot(data=df3, x="x", y="y", fill=True, cmap="copper",)
sns.kdeplot(data=df4, x="x", y="y", fill=True,  cmap="copper",)
sns.kdeplot(data=df5, x="x", y="y", fill=True,  cmap="copper",)
sns.kdeplot(data=df6, x="x", y="y", fill=True, cmap="copper",)
sns.kdeplot(data=df7, x="x", y="y", fill=True, cmap="copper",)
sns.kdeplot(data=df8, x="x", y="y", fill=True,  cmap="copper",)
sns.kdeplot(data=df9, x="x", y="y", fill=True,  cmap="copper",)
sns.kdeplot(data=df10, x="x", y="y", fill=True, cmap="copper",)
sns.kdeplot(data=df11, x="x", y="y", fill=True, cmap="copper",)
sns.kdeplot(data=df12, x="x", y="y", fill=True,  cmap="copper",)
sns.kdeplot(data=df13, x="x", y="y", fill=True,  cmap="copper",)
sns.kdeplot(data=df14, x="x", y="y", fill=True, cmap="copper",)
sns.kdeplot(data=df15, x="x", y="y", fill=True,  cmap="copper",)
sns.kdeplot(data=df16, x="x", y="y", fill=True,  cmap="copper",)
sns.kdeplot(data=df17, x="x", y="y", fill=True, cmap="copper",)
sns.kdeplot(data=df18, x="x", y="y", fill=True, cmap="copper",)
sns.kdeplot(data=df19, x="x", y="y", fill=True,  cmap="copper",)
sns.kdeplot(data=df20, x="x", y="y", fill=True,  cmap="copper",)
sns.kdeplot(data=df21, x="x", y="y", fill=True, cmap="copper",)
sns.kdeplot(data=df22, x="x", y="y", fill=True, cmap="copper",)
sns.kdeplot(data=df23, x="x", y="y", fill=True,  cmap="copper",)
sns.kdeplot(data=df24, x="x", y="y", fill=True,  cmap="copper",)
sns.kdeplot(data=df25, x="x", y="y", fill=True, cmap="copper",)
sns.kdeplot(data=df26, x="x", y="y", fill=True,  cmap="copper",)
sns.kdeplot(data=df27, x="x", y="y", fill=True,  cmap="copper",)
sns.kdeplot(data=df28, x="x", y="y", fill=True, cmap="copper",)
sns.kdeplot(data=df29, x="x", y="y", fill=True, cmap="copper",)
sns.kdeplot(data=df30, x="x", y="y", fill=True,  cmap="copper",)
sns.kdeplot(data=df31, x="x", y="y", fill=True,  cmap="copper",)
sns.kdeplot(data=df32, x="x", y="y", fill=True, cmap="copper",)
sns.kdeplot(data=df33, x="x", y="y", fill=True, cmap="copper",)
sns.kdeplot(data=df34, x="x", y="y", fill=True, cmap="copper",)
# Add labels and title
plt.scatter(actual_dist_player,quantized_player,facecolors='none', edgecolors='r')
plt.xlabel('actual_dist')
plt.ylabel('gaussian_noise_dist')
plt.title('Player noisy dist')

# Show the plot
plt.show()
