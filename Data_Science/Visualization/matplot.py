#matplot
import numpy as np
import matplotlib.pyplot as plt


plt.plot(np.arange(7))
plt.ylabel('numbers')
plt.xlabel('indexes')
plt.title('data')
plt.show()


plt.plot([1,2,3,4],[5,6,7,8]) #first list goes to x axis second list goes to y value
plt.title('Matplot')
plt.xlabel('Numbers')
plt.ylabel('values')
plt.grid() #we gets the square brackets in the plot when we use grid
plt.show() 


#using by ocean dots 'ro' r means red color, o means ocean dots
plt.plot([1,2,3,4],[5,6,7,8],'bo') #first list goes to x axis second list goes to y value
plt.title('Matplot')
plt.xlabel('Numbers')
plt.ylabel('values')
plt.grid() #we gets the square brackets in the plot when we use grid
plt.show() 


a=(np.arange(7))
plt.plot(a,a**2,'r*', label='*red')
plt.plot(a,a**2.2,'bs', label='^blue')
plt.plot(a,a**2.4,'g^', label='^green')
plt.title('Matplot')
plt.xlabel('Numbers')
plt.ylabel('values')
plt.grid() ##we gets the square brackets in the plot when we use grid
plt.legend() #we we used legend the labels of different axis are printed in the graph
plt.show()

#line width
plt.plot(a,a**2,'r*', label='*red',linewidth=1)
plt.plot(a,a**2.2,'bs', label='^blue')
plt.plot(a,a**2.4,'g^', label='^green')
plt.title('Matplot')
plt.xlabel('Numbers')
plt.ylabel('values')
plt.grid() ##we gets the square brackets in the plot when we use grid
plt.legend() #we we used legend the labels of different axis are printed in the graph
plt.show()

#add the color line properties

import matplotlib.pyplot as plt
lines=plt.plot([1,2,3,4],[1,2,3,4],[1,2,5,7],[1,4,3,7]) #x1,y1,x2,y2
plt.setp(lines[0],color='r' ,linewidth=2.0 )
plt.setp(lines[1],color='g' ,linewidth=2.0 )
#plt.setp(lines[2],'color','g' ,'linewidth',2.0 )
plt.grid()


import numpy as np
plt.figure()
a=np.exp(np.arange(5))
b=np.square(np.arange(5))

plt.subplot(211) #2 rows,1 column,1st figure
plt.grid()
plt.plot(np.arange(5),a,'b-')

plt.subplot(212) #2 rows,1 column,second figure
plt.grid()
plt.plot(np.arange(5),b,'g-')

#figure1
plt.figure(1)
plt.subplot(342)
plt.grid()                                     
plt.plot(a,'ro')

plt.subplot(341)
plt.grid()
plt.plot(a**2,'g*')

plt.subplot(343)
plt.grid()
plt.plot(a**2.2,'b-')            

plt.subplot(344)
plt.grid()
plt.plot(a**2.2,'y+')

#figure 2
plt.figure(3)
plt.grid()
plt.plot(a,'r')



#figure 2
plt.figure(2)
plt.subplot(342)
plt.grid()
plt.plot(a,'r')

plt.subplot(341)
plt.grid()
plt.plot(a**2,'g')

plt.subplot(343)
plt.grid()
plt.plot(a**2.2,'b')

plt.subplot(344)
plt.grid()
plt.plot(a**2.2,'o')

"""colors symbols
b: blue
g: green
r: red
c: cyan
m: magenta
y: yellow
k: black
w: white
"""

#matplot lab

x,y=[1,2,3,4],[5,6,7,8]
fig=plt.figure()
axes=fig.add_axes([0.1,0.1,0.8,0.8]) #left,bottom,width,hight of the axis
axes.plot([1,2,3,4],[5,6,7,8])
axes.set_xlabel('sachin')
axes.set_ylabel('paddu')
axes.set_title('made for each other')


fig=plt.figure()
axes1=fig.add_axes([0.1,0.1,0.8,0.8])
axes2=fig.add_axes([.2,.6,.1,.2])
axes1.plot(x,y)
axes1.set_xlabel('sachin')
axes1.set_ylabel('paddu')
axes1.set_title('made for each other')
axes2.plot(y,x)
axes2.set_xlabel('sachin')
axes2.set_ylabel('paddu')
axes2.set_title('made for each other')


#sub plots
fig,axes=plt.subplots(nrows=1,ncols=2)
axes[0].plot(x,y)
axes[0].set_title('first plot')

axes[1].plot(x,y)
axes[1].set_title('second plot')
plt.tight_layout()


#figure sizes
fig=plt.figure(figsize=(5,2),dpi=75)#dot per pixel
ax=fig.add_axes([0,0,1,1])
ax.plot(x,y)

#to save the matplotlab pictures
fig.savefig('mypicure.jpg',dpi=200)


#legend
x=np.arange(5)
fig=plt.figure(figsize=(5,2),dpi=75)#dot per pixel
ax=fig.add_axes([0,0,1,1])
ax.plot(x,x**2,'r*',label='xsquare') #*,>  are markers
ax.plot(x,x**3,'g>',label='x-cube')
ax.grid()
ax.legend(loc=0) #best fit 0

#styles
x,y=[1,2,3,4],[5,6,7,8]

fig=plt.figure()
axes=fig.add_axes([0.1,0.1,0.8,0.8]) 
axes.plot(x,y,color='#FF8C00')

fig=plt.figure()
axes=fig.add_axes([0.1,0.1,0.8,0.8]) 
axes.plot(x,y,color='green',linewidth=2) #width of the line

fig=plt.figure()
axes=fig.add_axes([0.1,0.1,0.8,0.8]) 
axes.plot(x,y,color='green',linewidth=2,alpha=10) #alpha gives brightness of the line


fig=plt.figure()
axes=fig.add_axes([0.1,0.1,0.8,0.8]) 
axes.plot(x,y,color='green',linewidth=2,alpha=10,ls='--') #ls is the line style of the line #also give steps

fig=plt.figure()
axes=fig.add_axes([0.1,0.1,0.8,0.8]) 
axes.plot(x,y,color='green',linewidth=2,alpha=10,ls='--',marker='1',markersize=20)

fig=plt.figure()
axes=fig.add_axes([0.1,0.1,0.8,0.8]) 
axes.plot(x,y,color='green',linewidth=2,alpha=10,ls='--',marker='o',markersize=20,markerfacecolor='yellow',markeredgewidth=3,markeredgecolor='pink')


#pip install plotly
#pip install cufflinks

#ploty is an interactive visualization library
#cufflinks connects ploty with pandas   
#gives better visualization
#df.iplot