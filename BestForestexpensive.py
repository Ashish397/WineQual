import csv
import numpy
import pandas
import random
from scipy import stats

#initialise random seed
numpy.random.seed(0)

#rawdata import function
def dataimport (filename, filename2):
    raw_data = open((filename), "rt")
    raw_data2 = open((filename2), "rt")
    data = pandas.read_csv(raw_data, delimiter = ";")
    data2 = pandas.read_csv(raw_data2, delimiter = ";")
    data['Red'] = 1
    data2['Red'] = 0
    raw_data.close()
    raw_data2.close()
    data = pandas.concat([data,data2], ignore_index=True)
    cols = list(data.columns.values)
    save = cols[len(cols)-2]
    cols[len(cols)-2]=cols[len(cols)-1]
    cols[len(cols)-1]=save
    data = data[cols]
    data = data.sample(frac=1).reset_index(drop=True)
    del(data2)
    del(raw_data)
    del(raw_data2)
    del(cols)
    del(save)
    return data

#data split function
def datacleen (data,ratio):
    perm = numpy.random.permutation(data.index)
    train = data.iloc[perm[:int(ratio*len(data.index))]]
    test = data.iloc[perm[int(ratio*len(data.index)):]]
    train = train.reset_index(drop = 1)
    test = test.reset_index(drop = 1)
    del(perm)
    return train, test

#tree split function
def split (df, var, ind):
    left  = numpy.array
    right = numpy.array
    left = numpy.delete(left, 0)
    right = numpy.delete(right, 0)
    for spit in range (len(df)):
        if ((df.iat[spit, ind])<var):
            left = numpy.append(left, spit)
        else:
            right =numpy.append(right, spit)
    return left, right

def dbs(vt, df, d):
    h=int(len(vt)/2)
    vl = vt[:h]
    vr = vt[h:]
    del(vt)
    besti = -1
    bvar = -1
    ginil = 50000
    halb = 50000
    halfi=-1
    hvar=-1
    set = False

    for it1 in range (len(df.columns)-1):
        svm = [-1,-2]
        minn = df.min()[it1]
        add = (df.max()[it1] - minn)/d
        for it2 in range (d):
            var = minn + (it2*add)
            left, right = split(df,var,it1)
            gini = numpy.sum(multerr(df.iloc[left,(len(df.columns)-1)],vl))+numpy.sum(multerr(df.iloc[right,(len(df.columns)-1)],vr))
            hal = abs(len(left)-len(right))
            gini = gini+(hal/1000000)
            if (hal<halb):
                halfi=it1
                hvar = var
                halb=hal
            if ((gini == ginil) and (set == True)):
                svm[1]=var
            else:
                set = False
            if (gini<ginil):
                svm[0]=var
                besti=it1
                bvar = var
                ginil=gini
                set = True
            if ((svm[1]>svm[0]) and (not set)):
                bvar = (svm[0]+svm[1])/2
    del(svm)
    del(set)
    del(vl)
    del(vr)
    del(h)
    del(ginil)
    del(halb)
    del(minn)
    del(add)
    del(var)
    del(left)
    del(right)
    del(gini)
    del(hal)
    return besti, bvar, halfi, hvar

#error functions
def sqercol (ycol, ocol):
    col = ycol-ocol
    return numpy.dot(col,col)

def multerr(dfcol,outputs):
    errcol = pandas.DataFrame([])
    for it in range (len(outputs)):
        errcol[it]=abs(dfcol-outputs[it])
    errout = errcol.min(axis=1)
    del(errcol)
    return errout

#make functions
def treemaked(oord, output, df, disc, std):
    #PARAMETER
    if (len(df)<22) or (len(output)==1):
        return round((df.iloc[:,(len(df.columns)-1)].sum())/(len(df)))

    array=[0,0,0,0]
    bi, bvar, hi, hvar = dbs(output,df,disc)
    left, right = split(df, bvar, bi)

    whilecount = -1
    while ((len(left)==0)or(len(right)==0)):
            whilecount = whilecount + 1
            bi = round((len(df.columns)-1)*numpy.random.random())
            bvar = (df.max()[bi]-df.min()[bi])*numpy.random.random()
            left, right = split(df, bvar, bi)
            #PARAMETER
            if (whilecount>20):
                left,right = split(df,hvar,hi)

    dl = df.iloc[left,:]
    dr = df.iloc[right,:]
    dl = dl.reset_index(drop = 1)
    dr = dr.reset_index(drop = 1)
    array[0] = oord[bi]
    array[1] = bvar
    vh = int(len(output)/2)
    vr = output[vh:]
    vl = output[:vh]
    errstdl = multerr(dl.iloc[:,(len(dl.columns)-1)],vl).std()
    errstdr = multerr(dr.iloc[:,(len(dr.columns)-1)],vr).std()

    del(vh)
    del(whilecount)
    del(bi)
    del(bvar)
    del(hi)
    del(hvar)
    del(left)
    del(right)

    if (errstdl<(std/6)):
        del(errstdl)
        array[2] = treemaked(oord,vl,dl,disc,std)
        del(vl)
    else:
        del(errstdl)
        del(vl)
        array[2] = treemaked(oord,output,dl,disc,std)

    if (errstdr<(std/6)):
        del(errstdr)
        array[3] = treemaked(oord,vr,dr,disc,std)
        del(vr)
    else:
        del(errstdr)
        del(vr)
        array[3] = treemaked(oord,output,dr,disc,std)

    del(dl)
    del(dr)
    return array

#read functions
def exploret(array, df):
    out=df.iloc[:,0].astype(int)
    for ite in range (len(df)):
        out.iat[ite]=explorec(array,df.iloc[ite,:])
    return out

def explorec(array,df):
    val = 0
    if((type(array))==float):
        return array
    elif (df.iat[array[0]] < array[1]):
        if (type(array[2])==list):
            val = explorec (array[2],df)
        else:
            val = array[2]
    else:
        if (type(array[3])==list):
            val = explorec (array[3],df)
        else:
            val = array[3]
    return int(val)

#Random Forest Functions
def randomforest (output, df, treenum, varnum, disc, std):
    order = [0,1,2,3,4,5,6,7,8,9,10,11]
    arrdf = pandas.Series([[]])
    for ti in range (treenum):
        print('start ', ti)
        numpy.random.shuffle(order)
        oord = numpy.append(order[0:varnum],len(df.columns)-1)
        arrdf = arrdf.append(pandas.Series([treemaked (oord, output, df.iloc[:,oord], disc, std)]))
    del(order)
    del(oord)
    return arrdf.iloc[1:]

def rfoutt (arrarr, df):
    val = exploret(arrarr.iat[0],df)
    for arit in range (1,len(arrarr)):
        val = val + exploret(arrarr.iat[arit],df)
    return val/len(arrarr)

#Initialise data for function
filename = "winequality-red.csv"
filename2 = "winequality-white.csv"

#get data
data = dataimport (filename, filename2)
#split data
trainb, test = datacleen(data, 0.8)
numpy.random.seed()
train, trainv = datacleen(trainb, 0.6)

#main func
outputs = []
minnn = int(data.min()[len(data.columns)-1])
maxxx = int(data.max()[len(data.columns)-1])

for oite in range (maxxx-minnn):
    outputs.append(minnn+oite)

besterror = 50000
for vni in range (1,(len(test.columns)-1)):
    for stdi in range (20,41):
        outarrdf = randomforest (outputs, train, 100, vni, 100, stdi/10)
        oarr = rfoutt (outarrdf, trainv)
        ycol = trainv.iloc[:,(len(trainv.columns)-1)]
        error = sqercol (ycol,oarr)
        print ('Error on ', vni, ' variables and ', stdi,' std: ', error)
        if (error<besterror):
            bestarrdf = outarrdf
            besterror = error
        print('Best error so far: ', besterror, ' for parameters: ', stdi, 'std and ', vni, ' variables')

ocol = rfoutt (bestarrdf, test)
ycol = test.iloc[:,(len(test.columns)-1)]
error = sqercol (ycol,ocol)
bias = numpy.mean(ycol-ocol)
print ('The test error on the random forest is:',(error/len(test)),'and the bias is: ', bias)
