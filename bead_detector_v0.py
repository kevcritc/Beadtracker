#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 19:36:30 2022

@author: phykc
"""
import cv2 as cv2
from math import pi,cos, tan, sin, sqrt
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from tkinter import *
from tkinter import filedialog

class Movie():  
    def __init__(self, path, file,wait_time, area_limit, blurvalue=3):
        """Reads the details of the movie file"""
        self.file=file
        self.path=path
        self.filename=os.path.join(self.path, self.file)
        self.starttime=time.time()
        self.cap=cv2.VideoCapture(self.filename)
        self.length=self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
        self.backgroundframe=0
        self.framerate=1
        self.blurvalue=blurvalue
        self.backSub = cv2.createBackgroundSubtractorMOG2()
        self.kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        self.wait_time=wait_time
        self.area_limit=area_limit
        
    def run(self):
        frameno=0
        self.frames=[]
        while frameno<self.length:
        #Read each frame ret is True if there is a frame
            ret, image = self.cap.read()
            
            #This end the movie when no frame is present
            if not ret:
                print('End of frames')
                cv2.waitKey(1)
                break
            if len(image.shape)==3:
                frame=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            else:
                frame=image.copy()
            
            fgMask = self.backSub.apply(frame)
            filterimg=cv2.GaussianBlur(fgMask, (self.blurvalue,self.blurvalue),3)
            hr,thresh = cv2.threshold(filterimg,100,255,cv2.THRESH_BINARY)
            erode=cv2.erode(thresh,self.kernel)
            cv2.imshow('show',erode)
            cv2.waitKey(wait_time)
                
            self.frames.append(Frame(erode,image,frameno, self.area_limit))
            frameno+=1
        cv2.destroyAllWindows()
        cv2.waitKey(1)

class Frame:
    def __init__(self, img,orimg, frameno, arealimit):
        self.beads=[]
        self.img=img
        self.imgcopy=self.img.copy()
        self.frameno=frameno
        self.orimg=orimg
        self.arealimit=arealimit
        contours, hierarchy = cv2.findContours(self.img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt)>1:
                x,y,w,h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                if area<self.arealimit:
                    cx=x+0.5*w
                    cy=y+0.5*h
                    self.beads.append(Bead(cx,cy,area, self.frameno, len(self.beads)))
                    cv2.rectangle(self.orimg,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow('found',self.orimg)
        cv2.waitKey(1)
class Bead:
    def __init__(self,cx,cy,area,frameno, number):
        self.x=cx
        self.y=cy
        self.area=area
        self.frame_no=frameno
        self.bead_no=number
        
class Process_beads(Movie):
    def __init__(self,path, file, scale,framerate, wait_time, area_limit):
        super().__init__(path, file, wait_time, area_limit)
        self.scale=scale
        self.tpf=framerate
        
    def create_data_lists(self):
        self.frame_no_list=[]
        self.x_list=[]
        self.y_list=[]
        self.area_list=[]
        self.bead_list=[]
       
        for no in range(len(self.frames)):
            for bead_info in self.frames[no].beads:
                self.frame_no_list.append(bead_info.frame_no)
                self.x_list.append(bead_info.x)
                self.y_list.append(bead_info.y)
                self.area_list.append(bead_info.area)
                self.bead_list.append(bead_info.bead_no)
        
    def pairup(self,Xmean_array, Ymean_array, size,maxdistance=800, w1=1,w2=1):
        self.size=size
        self.w1=w1
        self.w2=w2
        self.Xmean_array=Xmean_array
        self.Ymean_array=Ymean_array
        
        self.maxdistance=maxdistance
        framesl=np.arange(0,max(self.frame_no_list)+1)
        #Establish list of the indicies for cells tracked in each frame
        frameindexs=[]
        for frame in framesl:
            indexingframe=[]
            for i in range(len(self.frame_no_list)):
                if self.frame_no_list[i]==frame:
                    indexingframe.append(i)
            frameindexs.append(indexingframe)
        #Compare frame n with frame n+1 make smallest separations match, unless greater than 
        #a maxium to create a link
        self.frame1index=[]
        self.frame2index=[]
        # print(self.Xmean_array)
        for a in range(1,len(frameindexs)-1):
            #b and c become the IDs from the CSV file
            if len(frameindexs[a])>0 and len(frameindexs[a+1])>0:
             #Create an array to store the seprations from from fram n and n+1
                 separr=np.zeros((len(frameindexs[a]), len(frameindexs[a+1])),dtype=float)
                 for b1,b in enumerate(frameindexs[a]):
                 
                    for c1,c in enumerate(frameindexs[a+1]):
                        #find the separation for each particle and place them in an array
                        area1=self.area_list[b]
                        area2=self.area_list[c]
                        x1=self.x_list[b]
                        y1=self.y_list[b]
                        x2=self.x_list[c]
                        y2=self.y_list[c]
                        ix=int(x1//(self.width/self.size[0]))
                        jx=int(y1//(self.height/self.size[1]))
                    
                        seperation=self.w1*sqrt(((x2-self.Xmean_array[ix,jx])-x1)**2+((y2-self.Ymean_array[ix,jx])-y1)**2)+self.w2*abs(area2-area1)
                        
                        separr[b1,c1]=seperation
                 noresult=True
                 
                 for i in range(b1+1):
                    rep=0
                    noresult=True
                    while noresult==True and rep<2:
                        minval=np.amin(separr[i,:])
                        result=(np.where(separr[i,:] == np.amin(separr[i,:])))
                     
                        if minval==np.amin(separr[i:,result]) and minval<=self.maxdistance:
                            self.frame1index.append(frameindexs[a][i])
                            self.frame2index.append(frameindexs[a+1][result[0][0]])
                            noresult=False
                            
                            
                        else:
                            separr[i,result]=500
                            
                        rep+=1
    def listup(self):    
        #Create a list of the linking indicies between frame na dn n+1
        linklist=[]
        linkedup=[]
        for d in range(len(self.frame_no_list)):
            if d in self.frame1index:
                indy=self.frame1index.index(d)
                linklist.append(self.frame2index[indy])
            else:
                linklist.append('none')
            linkedup.append(False)
                
        #Create a set of trijectories index lists
        self.particlelist=[]
        
        for q in range(len(self.frame_no_list)):
            train=[]
            
            if linkedup[q]==False:
                train.append(q)
                
                linkedup[q]=True
                z=linklist[q]
                while z!='none' and linkedup[z]==False:
                    
                    train.append(z)
                    linkedup[z]=True
                    z=linklist[z]
                self.particlelist.append(train)
    def bead_pathways(self):    
        self.beadpaths=[]
        for bead in self.particlelist:
            numbersteps=len(bead)
            path=[]
            if numbersteps>1:
                for ID in bead:
                    #time in frames frame 1 in 0.
                    time=self.frame_no_list[ID]
                    x=self.x_list[ID]
                    y=self.y_list[ID]
                    area=self.area_list[ID]
                    cn=self.bead_list[ID]
                    
                    path.append([x,y,time,area,cn])
                self.beadpaths.append(path)
    def createdatapack(self): 
        self.datapack=[]        
        for track in self.beadpaths:
            steps=len(track)
            tpos=[]
            xpos=[]
            ypos=[]
            area=[]
            b_number=[]
            
            #loop through the cell paths 'track' and get the postions from frame n and n+1 for the calculations
            #possibly missing the very last points from x,y,t,r,inten data??
            for n in range(steps):
                
                x1=track[n][0]
                y1=track[n][1]
                t1=track[n][2]
                area.append(track[n][3])
                b_number.append(track[n][4])
                tpos.append(t1)
                xpos.append(x1)
                ypos.append(y1)
        
            #Put all the data in this list such that each path is an object in the list   
            self.datapack.append([xpos, ypos, tpos, area,b_number])    
        maxvalueallowed=self.maxdistance
        
        i=0
        while i<len(self.datapack)-1:
            #check the last time point and then look to see if a new time point starts +2 which is in the same region and has same colour.
            endt=self.datapack[i][2][-1]
            endx=self.datapack[i][0][-1]
            endy=self.datapack[i][1][-1]
            endarea=self.datapack[i][3][-1]
        
            matse=[]
            svals=[]
            for j in range(i+1,len(self.datapack)):
                if self.datapack[j][2][0]==endt+2:
                    
                    matse.append(j)
            
            if len(matse)>0:
            
                
                for v in matse:
                    sx=self.datapack[v][0][0]
                    sy=self.datapack[v][1][0]
                    sarea=self.datapack[v][3][0]
                    ix=int(sx//(self.width/self.size[0]))
                    jx=int(sy//(self.height/self.size[1]))
                    sep=self.w1*sqrt(((sx-self.Xmean_array[ix,jx]*2)-endx)**2+((sy-self.Ymean_array[ix,jx]*2)-endy)**2)+self.w2*abs(sarea-endarea)

                    svals.append(sep)
                   
                min_value=min(svals)
                min_index=svals.index(min_value)
                # This patches the 'the before and after' missed frame together
                if min_value<maxvalueallowed:
                    
                    for q in range(5):
                        a=self.datapack[i][q]
                        b=self.datapack[matse[min_index]][q]
                            
                        c=a+b
                        self.datapack[i][q]=c
                    self.datapack.pop(matse[min_index])
            i+=1
    def find_means(self):
        meansofx=[]
        meansofy=[]
        maxlength=0
        for data in self.datapack:
            if len(data)>3:
                xarray=np.array(data[0])
                yarray=np.array(data[1])
                xdiff=np.diff(xarray)
                ydiff=np.diff(yarray)
                mean_X, mean_Y= np.mean(xdiff),np.mean(ydiff)
                meansofx.append(mean_X)
                meansofy.append(mean_Y)
                if len(data)>maxlength:
                    maxlength=len(data)
        Xarry=np.array(meansofx)
        Yarry=np.array(meansofy)
        Xmean=np.mean(Xarry)
        Ymean=np.mean(Yarry)
        number_tracks=len(self.datapack)
        return Xmean, Ymean, number_tracks, maxlength
    def plotalltracks(self):
        for data in self.datapack:
            ax = plt.gca()
            ax.plot(data[0],data[1], linewidth=1.0)
            ax.set_xlabel('x /pixel')
            ax.set_ylabel('y /pixel')
        
           
        
        ax.invert_yaxis()    
        plt.show()
    def applyscales(self):
        
        #Unpack datapack into a full dataframe for one file at a time.
        # data pack : [xpos, ypos, tpos]
        
        for i,dat in enumerate(self.datapack):
            pathlength=len(dat[0])
            for j in range(pathlength):
                
                if i==0 and j==0:
                    d={'Bead number':[i],'x':[dat[0][j]],'y':[dat[1][j]],'t':[dat[2][j]], 'area':[dat[3][j]]}
                    df1=pd.DataFrame(data=d)
                
                elif  i>0 and j==0:
                    d={'Bead number':[i],'x':[dat[0][j]],'y':[dat[1][j]],'t':[dat[2][j]], 'area':[dat[3][j]]}
                    df2=pd.DataFrame(data=d)
                    df1=df1.append(df2,ignore_index = True)
                    
                elif j==1:
                    separation=sqrt((dat[0][j]-dat[0][j-1])**2+(dat[1][j]-dat[1][j-1])**2)
                    speed=separation/(dat[2][j]-dat[2][j-1])
                    deltax=dat[0][j]-dat[0][j-1]
                    deltay=dat[1][j]-dat[1][j-1]
                        
                    d={'Bead number':[i],'x':[dat[0][j]],'y':[dat[1][j]],'t':[dat[2][j]], 'area':[dat[3][j]],'di':[separation], 'vi':[speed], 'delx':[deltax], 'dely':[deltay]}
                    df2=pd.DataFrame(data=d)
                    df1=df1.append(df2,ignore_index = True)
                    
                elif j==2:
                    separation=sqrt((dat[0][j]-dat[0][j-1])**2+(dat[1][j]-dat[1][j-1])**2)
                    speed=separation/(dat[2][j]-dat[2][j-1])
                    deltax=dat[0][j]-dat[0][j-1]
                    deltay=dat[1][j]-dat[1][j-1]
                    d={'Bead number':[i],'x':[dat[0][j]],'y':[dat[1][j]],'t':[dat[2][j]], 'area':[dat[3][j]],'di':[separation], 'vi':[speed],'delx':[deltax], 'dely':[deltay]}
                    df2=pd.DataFrame(data=d)
                    df1=df1.append(df2,ignore_index = True)
                    
                elif j==pathlength-1:
                    separation=sqrt((dat[0][j]-dat[0][j-1])**2+(dat[1][j]-dat[1][j-1])**2)
                    speed=separation/(dat[2][j]-dat[2][j-1])
                    deltax=dat[0][j]-dat[0][j-1]
                    deltay=dat[1][j]-dat[1][j-1]
                    total_dis=separation+df1.loc[df1['Bead number'] == i, 'di'].sum()
                    net_dis=sqrt((dat[0][j]-dat[0][0])**2+(dat[1][j]-dat[1][0])**2)
                    total_time=(dat[2][j]-dat[2][0])
                    
                    selection=df1[df1['Bead number']==i]
                    length=len(selection)
                    xdata=selection['x'].tolist()
                    ydata=selection['y'].tolist()
                    dmax=0.0
                    for n in range(1,length):
                        d=sqrt((xdata[n]-xdata[0])**2+(ydata[n]-ydata[0])**2)
                        if d>dmax:
                            dmax=d
                    
                    max_dis=df1.loc[df1['Bead number'] == i, 'di'].max()

                   
                    mean_curv_speed=(speed+df1.loc[df1['Bead number'] == i, 'vi'].sum())/(pathlength-1)
                    deltanetx=dat[0][j]-dat[0][0]
                    deltanety=dat[1][j]-dat[1][0]
                    mean_sl_speed=net_dis/total_time
                    d={'Bead number':[i],'x':[dat[0][j]],'y':[dat[1][j]],'t':[dat[2][j]], 'area':[dat[3][j]],'di':[separation], 'vi':[speed],'delx':[deltax], 'dely':[deltay], 'Total Distance':[total_dis],'Net Distance':[net_dis],'Max Step':[max_dis],'dmax':[dmax],'Total Time':[total_time],'Mean straight-line speed':[mean_sl_speed],'Linear Forward Progression':[mean_sl_speed/mean_curv_speed],'net delta x':deltanetx,'net delta y':deltanety}
                    df2=pd.DataFrame(data=d)
                    df1=df1.append(df2,ignore_index = True)
                    
                else:
                    separation=sqrt((dat[0][j]-dat[0][j-1])**2+(dat[1][j]-dat[1][j-1])**2)
                    speed=separation/(dat[2][j]-dat[2][j-1])
                    deltax=dat[0][j]-dat[0][j-1]
                    deltay=dat[1][j]-dat[1][j-1]
                
                    d={'Bead number':[i],'x':[dat[0][j]],'y':[dat[1][j]],'t':[dat[2][j]], 'area':[dat[3][j]],'di':[separation], 'vi':[speed],'delx':[deltax], 'dely':[deltay]}
                    
                    df2=pd.DataFrame(data=d)
                    df1=df1.append(df2,ignore_index = True)
    
        df1['x_pix']=df1['x']
        df1['y_pix']=df1['y']
        df1['x']=df1['x']/self.scale
        df1['y']=df1['y']/self.scale
        df1['t']=df1['t']*self.tpf
        df1['area']=df1['area']/self.scale**2
        df1['di']=df1['di']/self.scale
        df1['vi']=df1['vi']/(self.scale*self.tpf)
        if df1.shape[1]>10:
            df1['Total Distance']=df1['Total Distance']/self.scale
            df1['Net Distance']=df1['Net Distance']/self.scale
            df1['Max Step']=df1['Max Step']/self.scale
            df1['dmax']=df1['dmax']/self.scale
            df1['Total Time']=df1['Total Time']*self.tpf
            df1['Mean straight-line speed']=df1['Mean straight-line speed']/(self.scale*self.tpf)
            df1['Net x velocity']=df1['net delta x']/df1['Total Time']
            df1['Net y velocity']=df1['net delta y']/df1['Total Time']
        
        
        self.dataframe=df1    
    def findlocalmeans(self, meanx, meany, size):
        self.size=size
        self.avearrayx=np.ones(self.size,dtype=float)
        self.avearrayy=np.ones(self.size,dtype=float)
        self.avearrayx*=meanx
        self.avearrayy*=meany
        
        xstep=self.width/self.size[0]
        ystep=self.height/self.size[1]
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                box=self.dataframe[(self.dataframe['x_pix']>i*xstep) & (self.dataframe['x_pix']<i*xstep+xstep) &(self.dataframe['y_pix']>j*ystep) & (self.dataframe['y_pix']<j*ystep+ystep)]
                if len(box)>0:
                    self.avearrayx[i,j]=box['delx'].mean()
                    self.avearrayy[i,j]=box['dely'].mean()
                else:
                    self.avearrayx[i,j]=meanx
                    self.avearrayy[i,j]=meany
                    
        return self.avearrayx,self.avearrayy
                    
                
        
        
    def write_excel(self):
        if self.dataframe.shape[1]>8:
            ave=self.dataframe['Mean straight-line speed'].mean()
            sem=self.dataframe['Mean straight-line speed'].sem()
            print(f'The mean speed is {ave:.0f} +/- {sem:.0f}')
        
        csvfilename=self.filename[:-4]+'_processed.xlsx'
        self.dataframe.to_excel(csvfilename)
        print('.xlsx File created')
    def plothisogram(self):
        vilist=self.dataframe['vi'].to_list()
        plt.hist(vilist)
        plt.ylabel('Frequency')
        plt.xlabel(r'Speed /µms$^{-1}$')
        plt.show()
     
class Beadtracker():
    def __init__(self,path, file, scale, framerate, guessx, guessy, wait_time, area_limit,size):
        self.size=size
        analysis=Process_beads(path, file, scale, framerate, wait_time, area_limit)
        analysis.run()
        # try:
        analysis.create_data_lists()
        x_arraymean=np.ones(self.size,dtype=float)
        y_arraymean=np.ones(self.size,dtype=float)
        x_arraymean*=guessx
        y_arraymean*=guessy
        
        analysis.pairup(x_arraymean,y_arraymean, self.size)
        analysis.listup()
        analysis.bead_pathways()
        analysis.createdatapack()
        xmean,ymean, tracks, maxlength=analysis.find_means()
        analysis.applyscales()
        x_arraymean, y_arraymean=analysis.findlocalmeans(xmean,ymean,self.size)
        deltaxy=50
        count=0
        while deltaxy>0.1 and count<25:
            pxmean,pymean=xmean,ymean
            maxdis=0.5*sqrt(pxmean**2+pymean**2)
            if maxdis<50:
                maxdis=100
            print(xmean,ymean, tracks, maxlength)
            analysis.pairup(x_arraymean,y_arraymean, self.size, maxdis)
            analysis.listup()
            analysis.bead_pathways()
            analysis.createdatapack()
            analysis.applyscales()
            xmean,ymean, tracks, maxlength=analysis.find_means()
           
            x_arraymean,y_arraymean=analysis.findlocalmeans(xmean,ymean,self.size)
            deltaxy=sqrt((xmean-pxmean)**2+(ymean-pymean)**2)
            count+=1
        try:
            analysis.plotalltracks()
            analysis.applyscales()
            analysis.write_excel()
            analysis.plothisogram()
        except ValueError:
             print('Failed to work!')


path=r'/Users/phykc/Documents/Work/beads/'    
file=r'D1 F X10 300UM V3.avi'
scale=1.55
framerate=0.44
guessx=100 #in pixels per frame
guessy=0 #in pixels per frame
wait_time=1 #in ms
area_limit=500 #in pixels
size=(1,100)
bt=Beadtracker(path, file, scale, framerate, guessx,guessy, wait_time, area_limit, size)




