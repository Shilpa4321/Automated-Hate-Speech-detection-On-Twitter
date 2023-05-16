from tkinter import *
from  tkinter import ttk
import pandas as pd

ws  = Tk()
ws.title('Hate speech')
ws.geometry('1000x400')
ws['bg'] = '#AC99F2'

data_frame = Frame(ws)
data_frame.pack()
b=40.02
data_frame = ttk.Treeview(data_frame)

data_frame['columns'] = ('id', 'text')

data_frame.column("#0", width=0,  stretch=YES)
data_frame.column("id",anchor=CENTER, width=80)
data_frame.column("text",anchor=CENTER,width=980)

data_frame.heading("#0",text="",anchor=CENTER)
data_frame.heading("id",text="Id",anchor=CENTER)
data_frame.heading("text",text="Name",anchor=CENTER)

df=pd.read_csv('results.csv')
for i in range(len(df)):
	data_frame.insert(parent='',index='end',iid=i,text='',values=(i,df['text'][i]))
data_frame.pack()

ws.mainloop()