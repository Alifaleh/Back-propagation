from random import random
import math

round_var=4

def sigmoid(x,i1,i2,ll):
    if i1==ll-1:
        print("o_z" + str(i2 + 1) + " = " + "1/(1+exp(-1*" + str((x+0.8)) + ")) = " + str(round(1 / (1 + math.exp(-1 * (x+0.8))), round_var)))
        return (round(1 / (1 + math.exp(-1 * (x+0.8))), round_var))
    else:
        print("h"+str(i1+1)+"_z"+str(i2+1)+" = "+"1/(1+exp(-1*"+str(x)+")) = "+str(round(1/(1+math.exp(-1*x)),round_var)))
        return(round(1/(1+math.exp(-1*x)),round_var))

def b_sigmoid(x,i1,i2,ll):
    if i1 == ll - 1:
        print("o_z" + str(i2 + 1) + " = " + "(2/(1+exp(-1*" + str(x) + ")))-1 = " + str(round((2 / (1 + math.exp(-1 * x))) - 1, round_var)))
    else:
        print("h"+str(i1+1)+"_z"+str(i2+1)+" = "+"(2/(1+exp(-1*" + str(x) + ")))-1 = "+str(round((2/(1+math.exp(-1*x)))-1,round_var)))
    return(round((2/(1+math.exp(-1*x)))-1,round_var))

def hl(x,i1,i2,ll):
    y=0
    if x>=0:
        y=1
    else:
        y=0
    if i1 == ll - 1:
        print("o_z" + str(i2 + 1) + " = HL(" + str(x) + ") = " + str(y))
    else:
        print("h" + str(i1 + 1) + "_z" + str(i2 + 1) + " = HL("+str(x)+") = "+str(y))
    return y

def b_hl(x,i1,i2,ll):
    y = 0
    if x >= 0:
        y = 1
    else:
        y = -1
    if i1 == ll - 1:
        print("o_z" + str(i2 + 1) + " = HL(" + str(x) + ") = " + str(y))
    else:
        print("h" + str(i1 + 1) + "_z" + str(i2 + 1) + " = HL(" + str(x) + ") = " + str(y))
    return y


def ht(x):
    print("ht = (1-exp(-2*" + str(x) + ")/(1+exp(-2*" + str(x) + ")) = " + str(round((1-math.exp(-2*x)) / (1 + math.exp(-2 * x)), round_var)))
    return round((1-math.exp(-2*x)) / (1 + math.exp(-2 * x)), round_var)


def dot_sum(x,y,v=True):
    str_f=""
    dp=[]
    _sum=0
    for i,c in enumerate(x):
        dp.append(round(c*y[i],round_var))
        str_f+="("+str(c)+"*"+str(y[i])+")+"
    str_f=str_f[:-1]
    _sum=sum(dp)
    str_f+=" = "+str(round(_sum,round_var))
    if v:
        print(str_f)
    else:
        print(str_f.split(" = ")[0],end="")
    return(round(_sum,round_var))


class NN:
    input=0
    layers=[]
    weights=[]
    n=0
    a=0
    af=[]
    def __init__(self,nofin,nofn,n,a,af):
        self.layers=nofn
        self.input=nofin
        self.n=n
        self.a=a
        self.af=af
        temp=[]
        temp.append(self.input)
        temp=temp+self.layers
        clw=[]
        for i,n in enumerate(temp):
            if i<len(temp)-1:
                tw=[]
                for x1 in range(0,temp[i+1]):
                    tw.append(0)
                for x2 in range(0,n):
                    clw.append(tw)
                self.weights.append(clw)
            clw=[]
    def use_rand_weights(self):
        self.weights=[]
        temp=[]
        temp.append(self.input)
        temp=temp+self.layers
        clw=[]
        for i,n in enumerate(temp):
            if i<len(temp)-1:
                tw=[]
                for x1 in range(0,temp[i+1]):
                    tw.append(random())
                for x2 in range(0,n):
                    clw.append(tw)
                self.weights.append(clw)
            clw=[]
    def use_weights(self,weights):
        self.weights=weights
    def predect(self,ip_r):
        print("\n\nPredecting : "+str(ip_r)+"\n")
        ip_r = [ip_r]
        nurons_results = []
        tl = []
        for nn in self.layers:
            for xx in range(0, nn):
                tl.append(0)
            nurons_results.append(tl)
            tl = []
        for i1, cl in enumerate(nurons_results):
            if i1 == 0:
                c_ip = ip_r
            else:
                c_ip = nurons_results[i1 - 1]
            for i2, n in enumerate(cl):
                if i1 == len(self.layers) - 1:
                    print("o_z" + str(i2 + 1) + "_ins = ", end="")
                else:
                    print("h" + str(i1 + 1) + "_z" + str(i2 + 1) + "_ins = ", end="")
                tcw = self.weights[i1]
                cw = []
                for i3, nw in enumerate(tcw):
                    for i4, w in enumerate(nw):
                        if i4 == i2:
                            cw.append(w)
                if self.af[i1] == "us":
                    nurons_results[i1][i2] = sigmoid(dot_sum(c_ip, cw), i1, i2, len(self.layers))
                elif self.af[i1] == "bs":
                    nurons_results[i1][i2] = b_sigmoid(dot_sum(c_ip, cw), i1, i2, len(self.layers))
                elif self.af[i1] == "hl":
                    nurons_results[i1][i2] = hl(dot_sum(c_ip, cw), i1, i2, len(self.layers))
                elif self.af[i1] == "bhl":
                    nurons_results[i1][i2] = b_hl(dot_sum(c_ip, cw), i1, i2, len(self.layers))
                elif self.af[i1] == "l":
                    nurons_results[i1][i2] = (dot_sum(c_ip, cw), i1, i2, len(self.layers))
        return nurons_results[-1]
    def train_on(self,ip,op,iterations):
        for it in range(0,iterations):
            print("For the iteration ("+str(it+1)+"):\n")
            print("The current weights are:")
            for i1,l in enumerate(self.weights):
                print("\nLayer(" + str(i1 + 1) + ") weights are:")
                for i2,n in enumerate(l):
                    for w in n:
                        if w>=0:
                            print(" "+str(w),end="  ")
                        else:
                            print(str(w), end="  ")
                    print("")
            for ib,ip_r in enumerate(ip):
                print("\nWhen the input is : "+str(ip_r))
                print("The output should be : " + str(op[ib])+"\n")
                print("\nForward Phase:\n")
                #Forward phase:
                nurons_results=[]
                tl=[]
                for nn in self.layers:
                    for xx in range(0,nn):
                        tl.append(0)
                    nurons_results.append(tl)
                    tl=[]
                for i1,cl in enumerate(nurons_results):
                    if i1==0:
                        c_ip=ip_r
                    else:
                        c_ip=nurons_results[i1-1]
                    for i2,n in enumerate(cl):
                        if i1==len(self.layers)-1:
                            print("o_z" + str(i2 + 1) + "_ins = ", end="")
                        else:
                            print("h"+str(i1+1)+"_z"+str(i2+1)+"_ins = ",end="")
                        tcw=self.weights[i1]
                        cw=[]
                        for i3,nw in enumerate(tcw):
                            for i4,w in enumerate(nw):
                                if i4==i2:
                                    cw.append(w)
                        if self.af[i1]=="us":
                            nurons_results[i1][i2] = sigmoid(dot_sum(c_ip, cw),i1,i2,len(self.layers))
                        elif self.af[i1]=="bs":
                            nurons_results[i1][i2] = b_sigmoid(dot_sum(c_ip, cw),i1,i2,len(self.layers))
                        elif self.af[i1]=="hl":
                            nurons_results[i1][i2] = hl(dot_sum(c_ip, cw),i1,i2,len(self.layers))
                        elif self.af[i1]=="bhl":
                            nurons_results[i1][i2] = b_hl(dot_sum(c_ip, cw),i1,i2,len(self.layers))
                        elif self.af[i1]=="l":
                            nurons_results[i1][i2] = dot_sum(c_ip, cw)
                    print("")

                print("\nError Calculation:\n")
                #Error calculation
                ole=[]
                hle=[]
                olr=nurons_results[-1]
                for i,r in enumerate(olr):
                    ole.append(round(r*(1-r)*(op[ib][i]-r),round_var))
                    print("o_e" + str(i + 1) + " = "+str(r)+"*(1-"+str(r)+")*("+str(op[ib][i])+"-"+str(r)+") = "+str(ole[-1]))
                nurons_results.reverse()
                rnr=list(nurons_results)[1:]
                nurons_results.reverse()
                for i1,l in enumerate(rnr):
                    cle=[]
                    for i2,n in enumerate(l):
                        nle=[]
                        if len(hle)==0:
                            nle=ole
                        else:
                            nle=hle[-1]
                        print("h" + str(len(self.layers)-(i1+1)) + "_e" + str(i2 + 1) +" = "+str(n)+"*(1-"+str(n)+")*",end="")
                        cle.append(round(n*(1-n)*dot_sum(nle, self.weights[(1+i1)*-1][i2],False),round_var))
                        print(" = "+str(cle[-1]))
                    hle.append(cle)
                hle.reverse()
                err=hle
                err.append(ole)
                ins=[ip_r]+nurons_results[:-1]
                #weights updating
                print("\nWeights Updating:\n")
                ow=list(self.weights)
                print(ow)
                for i1,l in enumerate(self.weights):
                    print("\nthe new layer("+str(i1+1)+") weights are:")
                    for i2,n in enumerate(l):
                        for i3,w in enumerate(n):
                            self.weights[i1][i2][i3]=round((self.n*err[i1][i3]*ins[i1][i2])+(self.a*ow[i1][i2][i3]),round_var)
                            if i1==len(self.layers)-1:
                                print("W_o(" + str(i2 + 1) + "," + str(i3 + 1) + ") = (" + str(self.n) + "*" + str(err[i1][i3]) + "*" + str(ins[i1][i2]) + ")+(" + str(self.a) + "*" + str(ow[i1][i2][i3]) + ") = " + str(self.weights[i1][i2][i3]))
                            else:
                                print("W_h" + str(i1+1)+"("+str(i2+1)+","+str(i3+1)+") = ("+str(self.n)+"*"+str(err[i1][i3])+"*"+str(ins[i1][i2])+")+("+str(self.a)+"*"+str(ow[i1][i2][i3])+") = "+ str(self.weights[i1][i2][i3]))






ip=[[0,0],[0,1],[1,0],[1,1]]
op=[[0],[0],[0],[1]]

nn=NN(2,[2,2,1],0.5,0.8,["us","us","us"])
nn.use_rand_weights()
nn.train_on(ip, op, 4)



