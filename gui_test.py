import tkinter as tk
from tkinter import*
from tkinter.filedialog import askdirectory
from btimage import BT_image, CellLabelOneImage, PrevNowCombo, TimeLapseCombo, Fov, WorkFlow , check_file_exist
from matplotlib import pyplot as plt
import tkinter.font as tkFont
from itertools import cycle
import time as time
import multiprocessing as mp
import sys
import os



#取得文件位置 #進行timelapsecomble
def selectpath():
    global newpath
    path_ = askdirectory()
    path.set(path_)
    newpath = path_.replace("/","\\")+"\\"
    print(newpath)

#進度條動畫
def pganimate(P):
    global progesszone
    for p in cycle(range(0,P)):
        if p == P-1 :
            progesszone.delete('label')
            window.update()
            time.sleep(0.5)
        elif p in [6,7,14,15,23,22]:
            continue
        else :
            progesszone.create_text(200, 125+18*p, text='V', font=ft1,tags=('label',))
            window.update()
            time.sleep(0.5)


def runtlc(newpath,STRATEGY,TARGET , SP_x,SP_y,BG_x,BG_y):
    TimeLapseCombo(root_path=newpath).combo(target=TARGET, save=True, strategy=STRATEGY, sp=(SP_x, SP_y), bg=( BG_x,BG_y))
    Fov(newpath).run()



def cellloi(newpath,current_target):
    global  after
    if current_target < 1:
        sys.stdout = open('file.txt', "w")
        print("please enter number bigger than one")
        sys.stdout.close()
    elif current_target == 1:
        picnum = 'first'
    else:
        picnum = 'old'
    after = CellLabelOneImage(newpath, target=current_target).run(adjust=True, plot_mode=False, load=picnum,save_water=True)


def pnc(newpath ,matchnum):
    if matchnum <= 1:
        sys.stdout = open('file.txt', "w")
        print("please enter number bigger than one")
        sys.stdout.close()
    else :
        output = PrevNowCombo(newpath).combo(now_target= matchnum, save=True)


#  多進程運行
def MpAni(s):
    if s == 7:
        pg_reconstruct = progesszone.create_text(200, 100, text='Phase Retrival', font=ft1)
        method_choose = int(retri_method.get())
        retrivalnum = int(retri_num.get())
        bg_x = int(BG_x.get())
        bg_y = int(BG_y.get())
        sp_x = int(SP_x.get())
        sp_y = int(SP_y.get())
        if method_choose == 2:
            print(BG_x)
            p1 = mp.Process(target=runtlc, args=(newpath, "cheat" , retrivalnum,sp_x,sp_y,bg_x,bg_y))
            p1.start()
        else :
            p1 = mp.Process(target=runtlc,args=(newpath,"try" , retrivalnum ,0,0,0,0))
            p1.start()

    elif s == 15:
        pg_reconstruct = progesszone.create_text(200, 100, text='Phase Retrival', font=ft1)
        pg_label = progesszone.create_text(200, 240, text='Cell Label', font=ft1)
        current_target = int(labelnum.get())
        p1 = mp.Process(target=cellloi , args = (newpath,current_target))
        p1.start()

    elif s == 23:
        pg_reconstruct = progesszone.create_text(200, 100, text='Phase Retrival', font=ft1)
        pg_label = progesszone.create_text(200, 240, text='Cell Label', font=ft1)
        pg_match = progesszone.create_text(200, 385, text='Figure Matching', font=ft1)
        matchnum = int(matchingnum.get())
        p1 = mp.Process(target=pnc, args=(newpath ,matchnum))
        p1.start()

    pganimate(s)

#messagebox 更新
def messa_upd():
    input = open("file.txt").read()
    message.set(input)
    msbox.after(50,messa_upd)

def messa_input():
    m_input = messainput.get()
    if m_input == "cat":
        f = open('cat.txt' ,"r")
        sys.stdout = open('file.txt', "w")
        print(f.read())
        sys.stdout.close()

    else :
        sys.stdout = open('file.txt',"w")
        print(m_input)
        print("press L to use input number as label ")
        sys.stdout.close()


if __name__ == "__main__":
    global current_target
    window = tk.Tk()
    window.title('Awesome GUI')
    window.geometry('1000x700')
    progesszone = Canvas(window, width=400, height=650)
    ft1 = tkFont.Font(family='Fixdsys', size=20, weight=tkFont.BOLD)

    path = tk.StringVar()
    Matchnum = tk.StringVar()
    Current_tar = tk.StringVar()
    retri_method = tk.IntVar()
    retri_num = tk.StringVar()
    SP_x = tk.IntVar()
    SP_y = tk.IntVar()
    BG_x = tk.IntVar()
    BG_y = tk.IntVar()
    message = tk.StringVar()
    messainput = tk.StringVar()


    #選擇路徑
    tk.Label(window,text ='Root Path: ').grid(row= 0 ,column = 0,sticky='w')
    tk.Button(window,text = 'select path',command = selectpath,width =10).grid(row = 0,column =8,sticky='w')
    tk.Entry(window,textvariable= path,width =40).grid(row = 0 , column = 1,columnspan = 7,sticky='w' )

    #進行timelapsecomble
    tk.Label(window,text =  'Phase Retrival: ').grid(row = 1 , column = 0,sticky='w')
    tk.Entry(window , textvariable = retri_num , width = 40 ).grid(row = 1,column =1,sticky="w",columnspan = 7 )
    tk.Button(window, text='Confirm', command=lambda: MpAni(7),width =10).grid(row=1, column=8,sticky='w')  # pad 可調整與邊緣距離
    tk.Radiobutton(window,text = "try" , variable = retri_method , value = 1 ).grid(row = 2,column =1,sticky="e")
    tk.Radiobutton(window, text="cheat", variable=retri_method, value=2).grid(row = 2,column =2,sticky="w")
    tk.Label(window,text = "SP   ( ").grid(row = 2, column = 3,sticky="n")
    tk.Label(window,text = " , ").grid(row = 2, column = 5,sticky="n")
    tk.Label(window, text=")").grid(row=2, column=7, sticky="n")
    tk.Label(window,text = "BG   ( ").grid(row = 3, column = 3,sticky="n")
    tk.Label(window,text = " , ").grid(row = 3, column = 5,sticky="n")
    tk.Label(window, text=")").grid(row=3, column=7, sticky="n")
    tk.Entry(window, textvariable=SP_x,width = 5).grid(row = 2, column = 4,sticky="n")
    tk.Entry(window, textvariable=SP_y,width = 5).grid(row = 2, column = 6,sticky="n")
    tk.Entry(window, textvariable=BG_x,width = 5).grid(row = 3, column = 4,sticky="n")
    tk.Entry(window, textvariable=BG_y,width = 5).grid(row = 3, column = 6,sticky="n")

    #進行label
    tk.Label(window,text ='Enter Label Number: ').grid(row= 4 ,column = 0,sticky='w')
    labelnum = tk.Entry(window,textvariable=Current_tar,width = 40)
    labelnum.grid(row= 4 ,column = 1,columnspan = 7,sticky='w' )
    tk.Button(window,text = 'Start Label',command =lambda : MpAni(15),width = 10).grid(row= 4 ,column = 8)

    #進行maching
    tk.Label(window,text ="Enter Match Number: ").grid(row= 5 ,column = 0,sticky='w')
    matchingnum = tk.Entry(window,textvariable=Matchnum,width = 40)
    matchingnum.grid(row= 5 ,column = 1,columnspan = 7,sticky='w' )
    tk.Button(window,text = 'maching',command = lambda : MpAni(23),width = 10).grid(row=5 ,column = 8 )

    #進度區塊
    progesszone.grid(row = 0,column = 9, rowspan = 23)
    title = progesszone.create_text(200,40,text = 'Step Progress Bar ',font = ft1)
    progesszone.create_rectangle(20,20,380,630)

    #文字區塊
    msbox = tk.LabelFrame(window,text = "MessageBox",width=500,height=450)
    msbox.grid(row=6,column=0,rowspan=15,columnspan = 9)
    msbox.grid_propagate(False)
    # msbox.grid_propagate(False)
    # canvas = tk.Canvas(msbox,width=400,height=400,scrollregion=(0,0,0,600))
    # vbar = tk.Scrollbar(msbox, orient="vertical")
    # vbar.pack(side = RIGHT ,fill =Y)
    # vbar.config(command = canvas.yview)
    # canvas.config(width=400,height=400)
    # canvas.configure(yscrollcommand = vbar.set)
    # canvas.pack()
    tk.Label(msbox, textvariable= message, justify=LEFT).grid(sticky = "w")
    msbox.after(50, messa_upd)

    #文字區塊 no 輸入
    tk.Label(window , text = "Message Box input zone : ").grid(row = 21 , column = 3,columnspan =4 )
    tk.Entry(window , textvariable = messainput,width = 10).grid(row = 21 , column = 7)
    tk.Button(window , text = "ok" , command = messa_input).grid(row = 21 , column = 8)


    window.mainloop()

