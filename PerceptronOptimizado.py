import numpy as np
from tkinter import *
from tkinter import ttk
import tkinter as tk
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
import statistics

matplotlib.use('TkAgg')


def importData():
    global X1, X2, Y

    # df = pd.read_csv(
    #     'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    # X1_1 = df.iloc[0:8, [0]].values
    # X2_1 = df.iloc[0:8, [2]].values
    # Y_1 = df.iloc[0:8, 4].values
    # X1_2 = df.iloc[50:58, [0]].values
    # X2_2 = df.iloc[50:58, [2]].values
    # Y_2 = df.iloc[50:58, 4].values

    # X1=np.concatenate((X1_1,X1_2))
    # X2=np.concatenate((X2_1,X2_2))
    # Y=np.concatenate((Y_1,Y_2))

    # Y = np.where(Y == 'Iris-setosa', -1, 1)
    # Y = np.squeeze(np.asarray(Y))

    # X1 = np.array([0, 1, 0, 1])
    # X2 = np.array([0, 0, 1, 1])
    # Y = np.array([-1, 1, 1, 1])

    X1 = np.array([3.9,4,3.8,2.5,1.6,3.9,3.9,2.9,3.2,3.4,2.3,1.6,4.6,4.3,3.1,2.5,3.4,1.6,2.7,2.3,2,1.8])
    X2 = np.array([1.8,2.3,2.3,2.1,1.1,3.9,2.5,1.3,1.7,3.6,2.5,1.7,3.4,3.1,1.8,2.9,4,1.9,1.5,1.4,2.7,3.6])
    Y = np.array([1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,1,-1,-1])

def preprocess():
    global normalized_X1, normalized_X2
    normalized_X1 = X1/np.linalg.norm(X1)
    normalized_X2 = X2/np.linalg.norm(X2)
    normalized_X1 = np.squeeze(np.asarray(normalized_X1))
    normalized_X2 = np.squeeze(np.asarray(normalized_X2))


def draw():
    global X1A, X2A, X1B, X2B

    indexYA = np.where(Y > 0)
    indexYB = np.where(Y <= 0)

    X1A = normalized_X1
    X2A = normalized_X2
    X1B = normalized_X1
    X2B = normalized_X2

    X1A = np.delete(X1A, indexYA)
    X2A = np.delete(X2A, indexYA)
    X1B = np.delete(X1B, indexYB)
    X2B = np.delete(X2B, indexYB)

    """-------------------- Graph of Y classes ---------------------------"""
    axes_train.plot(X1A, X2A, 'bx')
    axes_train.plot(X1B, X2B, 'ro')
    axes_train.grid(True)
    figure_train.canvas.draw()


"""------------------------------------------- Graph error ---------------------------"""
def graph_error(errorInput):
    axes_error.clear()
    axes_error.grid(True)
    axes_error.set_title("Error de iteración", fontsize=16)
    axes_error.plot(errorInput, 'r')
    figure_error.canvas.draw()
    figure_error.canvas.flush_events()


"""----------------------------------------- Graph slopes ---------------------------"""
def graph_slope1(x,slopeInput):
    axes_w1w2.clear()
    axes_w1w2.grid(True)
    axes_w1w2.set_title("-W1/W2", fontsize=14)
    axes_w1w2.plot(x,slopeInput, color='magenta')
    figure_w1w2.canvas.draw()
    figure_w1w2.canvas.flush_events()

def graph_slope2(x,slopeInput):
    axes_w0w2.clear()
    axes_w0w2.grid(True)
    axes_w0w2.set_title("-W0/W2", fontsize=14)
    axes_w0w2.plot(x,slopeInput, color='deeppink')
    figure_w0w2.canvas.draw()
    figure_w0w2.canvas.flush_events()


"""----------------------------------------- Graph weights ---------------------------"""
def graph_weights(w0,w1,w2):
    axes_w0.clear()
    axes_w1.clear()
    axes_w2.clear()
    axes_w0.grid(True)
    axes_w1.grid(True)
    axes_w2.grid(True)
    axes_w0.set_title("W0", fontsize=14)
    axes_w1.set_title("W1", fontsize=14)
    axes_w2.set_title("W2", fontsize=14)
    axes_w0.plot(w0, color='cyan')
    axes_w1.plot(w1, color='green')
    axes_w2.plot(w2, color='red')
    figure_w0.canvas.draw()
    figure_w1.canvas.draw()
    figure_w2.canvas.draw()
    figure_w1.canvas.flush_events()
    figure_w1.canvas.flush_events()
    figure_w2.canvas.flush_events()


def slopeInData(x,slope1,slope2):
    axes_train.clear()
    axes_train.grid(True)
    axes_train.set_title("Clases", fontsize=20)
    axes_train.plot(X1A, X2A, 'bx')
    axes_train.plot(X1B, X2B, 'ro')
    axes_train.plot(x, (x*slope1+slope2), color='lime')
    figure_train.canvas.draw()
    figure_train.canvas.flush_events()


def functionstop():
    global breakTrain
    breakTrain = True

def clear():
    axes_train.clear()
    axes_w0.clear()
    axes_w1.clear()
    axes_w2.clear()
    axes_w0w2.clear()
    axes_w1w2.clear()
    axes_error.clear()
    axes_error_epoca.clear()


def train():

    global breakTrain

    timeInitial = time.time()

    global theta, W

    if selectMod.get() == 2:  # ONLINE MOD

        """-------------------- Algorithm Perceptron --------------------"""

        # 1) Initialized weights
        w1 = random.uniform(-1,1)
        w2 = random.uniform(-1,1)
        theta = random.uniform(-1,1)  # threshold

        alpha = 0.5

        W = np.array([w1, w2])

        numbersAux = np.arange(min(normalized_X1)-max(normalized_X1)/10,
                               max(normalized_X1)+max(normalized_X1)/10, 0.01)

        numEpocas = 0

        ErrorEpoca = 0.8
        ErrorEpocaArray = []

        errorThreshold = 0.01

        while ErrorEpoca < -errorThreshold or ErrorEpoca > errorThreshold:

            if breakTrain:
                break

            errorY = []
            W0 = []
            W1 = []
            W2 = []
            slopeW0 = []
            slopeW1 = []

            for i in range(X1.shape[0]):
                W0.append(theta)
                W1.append(W[0])
                W2.append(W[1])

                # 2) Take inputs in i
                inputNeuron = np.array([normalized_X1[i], normalized_X2[i]])
                neti = np.dot(inputNeuron, W) + theta

                # 3) Calculate Y
                Yestimate = np.tanh(neti)
                errorY.append(Y[i]-Yestimate)

                # Calculate slopes
                slopeW0.append(-theta/W[1])
                slopeW1.append(-W[0]/W[1])

                if (errorY[i] > -0.1 or errorY[i] < 0.1):
                    # 4) Adaptation weights and threshold
                    W = W+alpha*errorY[i]*inputNeuron
                    theta = theta + errorY[i]
                else:
                    print("pase")
                    pass


            slopeInData(numbersAux,slopeW1[len(slopeW1)-1],slopeW0[len(slopeW0)-1])


            #graph_weights(W0,W1,W2)
            #graph_error(errorY)
            #graph_slope1(numbersAux, numbersAux *(-W1[len(W1)-1]/W2[len(W2)-1]))
            #graph_slope2(numbersAux, numbersAux *(-W0[len(W0)-1]/W2[len(W2)-1]))


            ErrorEpoca=0

            for i in range(len(errorY)):
                ErrorEpoca = ErrorEpoca+(errorY[i]**2)

            ErrorEpoca = 0.5*ErrorEpoca
            ErrorEpocaArray.append(ErrorEpoca)

            try:
                if statistics.pstdev(ErrorEpocaArray[len(ErrorEpocaArray)-10:len(ErrorEpocaArray)]) < 1:
                   alpha = 1
                else:
                   alpha = 0.5
            except:
                alpha = 0.5

            numEpocas = numEpocas+1

            """------------------------------------------- Graph epoch error ---------------------------"""
            axes_error_epoca.plot(ErrorEpocaArray, 'r')
            axes_error_epoca.grid(True)
            figure_error_epoca.canvas.draw()
            figure_error_epoca.canvas.flush_events()

    elif selectMod.get() == 1:  # OFFLINE MOD

        """-------------------- Algorithm Perceptron --------------------"""

        # 1) Initialized weights
        w1 = random.uniform(-1,1)
        w2 = random.uniform(-1,1)
        theta = random.uniform(-1,1)  # threshold

        alpha = 0.1

        W = np.array([theta,w1, w2])

        numbersAux = np.arange(min(normalized_X1)-max(normalized_X1)/10,
                               max(normalized_X1)+max(normalized_X1)/10, 0.01)

        numEpocas = 0

        ErrorEpoca = 0.8
        ErrorEpocaArray = []

        W0Epoca=[]
        W1Epoca=[]
        W2Epoca=[]

        errorThreshold = 0.01

        while ErrorEpoca < -errorThreshold or ErrorEpoca > errorThreshold:

            if breakTrain:
                break

            errorY = []
            W0 = []
            W1 = []
            W2 = []
            slopeW0 = []
            slopeW1 = []
            dW0=[]
            dW1=[]
            dW2=[]

            for i in range(X1.shape[0]):
                W0.append(W[0])
                W1.append(W[1])
                W2.append(W[2])

                # 2) Take inputs in i
                inputNeuron = np.array([1,normalized_X1[i], normalized_X2[i]])
                neti = np.dot(inputNeuron, W)

                # 3) Calculate Y
                Yestimate = np.tanh(neti)
                errorY.append(Y[i]-Yestimate)

                # Calculate slopes
                slopeW0.append(-W[0]/W[2])
                slopeW1.append(-W[1]/W[2])

                phi_p = (1-Yestimate**2)

                dW0.append(alpha*(errorY[i]*phi_p*inputNeuron[0]))
                dW1.append(alpha*(errorY[i]*phi_p*inputNeuron[1]))
                dW2.append(alpha*(errorY[i]*phi_p*inputNeuron[2]))




            dW0Sum=sum(dW0)
            dW1Sum=sum(dW1)
            dW2Sum=sum(dW2)

            W[0] = W[0]+(dW0Sum/len(dW0))
            W[1] = W[1]+(dW1Sum/len(dW1))
            W[2] = W[2]+(dW2Sum/len(dW2))

            W0Epoca.append(W[0])
            W1Epoca.append(W[1])
            W2Epoca.append(W[2])


            slopeInData(numbersAux,slopeW1[len(slopeW1)-1],slopeW0[len(slopeW0)-1])

            graph_weights(W0,W1,W2)
            graph_error(errorY)
            graph_slope1(numbersAux, numbersAux *(-W1[len(W1)-1]/W2[len(W2)-1]))
            graph_slope2(numbersAux, numbersAux *(-W0[len(W0)-1]/W2[len(W2)-1]))

            ErrorEpoca=0

            for i in range(len(errorY)):
                ErrorEpoca = ErrorEpoca+(errorY[i]**2)

            ErrorEpoca = 0.5*ErrorEpoca
            ErrorEpocaArray.append(ErrorEpoca)

            try:
                if statistics.pstdev(ErrorEpocaArray[len(ErrorEpocaArray)-len(X1):len(ErrorEpocaArray)]) < 0.1 and statistics.pstdev(ErrorEpocaArray[len(ErrorEpocaArray)-len(X1):len(ErrorEpocaArray)]) > 0.01:
                   alpha = 3
                elif statistics.pstdev(ErrorEpocaArray[len(ErrorEpocaArray)-10:len(ErrorEpocaArray)]) < 0.01 and ErrorEpoca < 4 and ErrorEpoca > 1.5:
                    alpha += 100
                elif statistics.pstdev(ErrorEpocaArray[len(ErrorEpocaArray)-10:len(ErrorEpocaArray)]) < 0.01 and ErrorEpoca < 1.5 and ErrorEpoca > 0.7:
                    alpha = 12
                elif statistics.pstdev(ErrorEpocaArray[len(ErrorEpocaArray)-10:len(ErrorEpocaArray)]) < 0.01 and ErrorEpoca < 0.7 and ErrorEpoca > 0.1:
                    alpha = 13
                else:
                   alpha = 2
            except:
                alpha = 2

            numEpocas = numEpocas+1

            """------------------------------------------- Graph epoch error ---------------------------"""
            axes_error_epoca.plot(ErrorEpocaArray, 'r')
            axes_error_epoca.grid(True)
            figure_error_epoca.canvas.draw()
            figure_error_epoca.canvas.flush_events()

    """-------------------------------------------Print data result ---------------------------"""

    txtTheta = tk.DoubleVar(value=round(theta, 3))
    txtW1 = tk.DoubleVar(value=round(W[0], 3))
    txtW2 = tk.DoubleVar(value=round(W[1], 3))
    txtNumEpocas = tk.IntVar(value=numEpocas)

    # Entries
    entryTheta = Entry(textvariable=txtTheta, justify=CENTER, width=8, fg='black', font=(
        'Calibri', 12), state=DISABLED).place(x=970, y=270)
    entryW1 = Entry(textvariable=txtW1, justify=CENTER, width=8, fg='black', font=(
        'Calibri', 12), state=DISABLED).place(x=1070, y=270)
    entryW2 = Entry(textvariable=txtW2, justify=CENTER, width=8, fg='black', font=(
        'Calibri', 12), state=DISABLED).place(x=1170, y=270)
    entryNumEpocas = Entry(textvariable=txtNumEpocas, justify=CENTER, width=3, fg='black', font=(
        'Calibri', 12), state=DISABLED).place(x=1290, y=270)

    # Labels
    LabelTheta = ttk.Label(text="Theta", background="#087E8B", font=(
        'Calibri', 12, 'bold')).place(x=980, y=245)
    LabelW1 = ttk.Label(text="W1", background="#087E8B", font=(
        'Calibri', 12, 'bold')).place(x=1090, y=245)
    LabelW2 = ttk.Label(text="W2", background="#087E8B", font=(
        'Calibri', 12, 'bold')).place(x=1190, y=245)
    LabelNumEpocas = ttk.Label(text="Epocas", background="#087E8B", font=(
        'Calibri', 12, 'bold')).place(x=1280, y=245)

    timeFinal = time.time()

    print("Tiempo de ejecución: ", timeFinal-timeInitial)


def functionStart():
    Ye = []

    if selectMod.get()==2:    #ONLINE MOD
        for i in range(X1.shape[0]):

            inputNeuron = np.array([normalized_X1[i], normalized_X2[i]])
            neti = np.dot(inputNeuron, W) + theta
            Yestimate = np.tanh(neti)

            Ye.append(Yestimate)

    elif selectMod.get()==1:  #OFFLINE MOD
        for i in range(X1.shape[0]):

            inputNeuron = np.array([1,normalized_X1[i], normalized_X2[i]])
            neti = np.dot(inputNeuron, W)
            Yestimate = np.tanh(neti)

            Ye.append(Yestimate)

    """------------------------------------------- Graph comparison ---------------------------"""
    axes_comparison.plot(Y, 'b',label='Datos reales')
    axes_comparison.plot(Ye, 'r',label='Datos estimados')
    axes_comparison.legend()
    figure_comparison.canvas.draw()
    figure_comparison.canvas.flush_events()



class Application(ttk.Frame):

    def __init__(self, main_window):
        super().__init__(main_window)

        global axes_error, axes_w0, axes_w1, axes_w2, axes_train, axes_w1w2, axes_w0w2, axes_comparison, axes_error_epoca
        global figure_train, figure_error, figure_w0, figure_w1, figure_w2, figure_w0w2, figure_w1w2, figure_comparison, figure_error_epoca

        main_window.title("IA Perceptron Application")
        main_window.configure(width=1350, height=600)
        main_window.eval('tk::PlaceWindow . center')
        self.place(relwidth=1, relheight=1)

        # Frames
        self.frm1 = Frame(self, height=600, width=1350,
                          background="#3C3C3C").place(x=0, y=0)
        self.frm2 = Frame(self, height=80, width=250,
                          background="#FF5A5F").place(x=10, y=10)
        self.frm3 = Frame(self, height=100, width=250,
                          background="#F5F5F5").place(x=10, y=100)
        self.frm4 = Frame(self, height=300, width=360,
                          background="#087E8B").place(x=270, y=10)
        self.frm5 = Frame(self, height=100, width=270,
                          background="#087E8B").place(x=10, y=210)
        self.frm6 = Frame(self, height=270, width=350,
                          background="#7B929A").place(x=10, y=320)
        self.frm7 = Frame(self, height=580, width=300,
                          background="#FF5A5F").place(x=640, y=10)
        self.frm8 = Frame(self, height=270, width=260,
                          background="#F78084").place(x=370, y=320)
        self.frm9 = Frame(self, height=300, width=390,
                          background="#087E8B").place(x=950, y=10)
        self.frm10 = Frame(self, height=270, width=390,
                           background="#7B929A").place(x=950, y=320)

        # Labels
        self.Label1 = ttk.Label(self, text="Mateo Pulido Aponte", background="#FF5A5F",
                                foreground='white', font=('Calibri', 14, 'bold')).place(x=40, y=20)
        self.Label2 = ttk.Label(self, text="Perceptron", background="#FF5A5F", foreground='black', font=(
            'Calibri', 14, 'bold')).place(x=80, y=50)

        # Buttons
        self.btn_import = tk.Button(self, width=20, text="Import", background="#ACFCD9", font=(
            'Calibri', 12, 'bold'), command=importData).place(x=30, y=110)
        self.btn_preprocess = tk.Button(self, width=20, text="Preprocess", background="#55D6BE", font=(
            'Calibri', 12, 'bold'), command=preprocess).place(x=30, y=150)
        self.btn_draw = tk.Button(self, width=20, text="Draw", background="#7FBAC0", font=(
            'Calibri', 12, 'bold'), command=draw).place(x=30, y=220)
        self.btn_train = tk.Button(self, width=12, text="Train", background="#3C3C3C", foreground='white', font=(
            'Calibri', 11, 'bold'), command=train).place(x=70, y=330)
        self.btn_function = tk.Button(self, width=12, text="Function", background="#087E8B", foreground='white', font=(
            'Calibri', 11, 'bold'), command=functionStart).place(x=1020, y=330)
        self.btn_stop = tk.Button(self, width=12, text="Stop", background="#087E8B", foreground='white', font=(
            'Calibri', 11, 'bold'), command=functionstop).place(x=210, y=330)
        self.btn_Clear = tk.Button(self, width=12, text="Clear", background="#3C3C3C", foreground='white', font=(
            'Calibri', 11, 'bold'), command=clear).place(x=1180, y=330)

        # Radio buttons
        self.button_modOFFLINE = tk.Radiobutton(self, text="OFFLINE", value=1, foreground='white', background="#087E8B", font=(
            'Calibri', 12, 'bold'), variable=selectMod).place(x=30, y=270)
        self.button_modONLINE = tk.Radiobutton(self, text="ONLINE", value=2, foreground='white', background="#087E8B", font=(
            'Calibri', 12, 'bold'), variable=selectMod).place(x=150, y=270)

        # Plots

        # Train
        figure_train = Figure(figsize=(6, 5), dpi=55)
        figure_train_canvas = FigureCanvasTkAgg(figure_train)
        axes_train = figure_train.add_subplot()
        figure_train_canvas.get_tk_widget().place(x=285, y=20)
        axes_train.set_title("Clases", fontsize=20)
        axes_train.grid(True)

        # Error
        figure_error = Figure(figsize=(6, 3.7), dpi=55)
        figure_error_canvas = FigureCanvasTkAgg(figure_error)
        axes_error = figure_error.add_subplot()
        figure_error_canvas.get_tk_widget().place(x=20, y=375)
        axes_error.set_title("Error de iteración", fontsize=16)
        axes_error.grid(True)

        # Wights

        # W0
        figure_w0 = Figure(figsize=(5.1, 3.3), dpi=55)
        figure_w0_canvas = FigureCanvasTkAgg(figure_w0)
        axes_w0 = figure_w0.add_subplot()
        figure_w0_canvas.get_tk_widget().place(x=650, y=20)
        axes_w0.set_title("W0", fontsize=14)
        axes_w0.grid(True)

        # W1
        figure_w1 = Figure(figsize=(5.1, 3.3), dpi=55)
        figure_w1_canvas = FigureCanvasTkAgg(figure_w1)
        axes_w1 = figure_w1.add_subplot()
        figure_w1_canvas.get_tk_widget().place(x=650, y=210)
        axes_w1.set_title("W1", fontsize=14)
        axes_w1.grid(True)

        # #W2
        figure_w2 = Figure(figsize=(5.1, 3.3), dpi=55)
        figure_w2_canvas = FigureCanvasTkAgg(figure_w2)
        axes_w2 = figure_w2.add_subplot()
        figure_w2_canvas.get_tk_widget().place(x=650, y=400)
        axes_w2.set_title("W2", fontsize=14)
        axes_w2.grid(True)

        # Slopes

        # -W1/W2
        figure_w1w2 = Figure(figsize=(4.4, 2.3), dpi=55)
        figure_w1w2_canvas = FigureCanvasTkAgg(figure_w1w2)
        axes_w1w2 = figure_w1w2.add_subplot()
        figure_w1w2_canvas.get_tk_widget().place(x=380, y=325)
        axes_w1w2.set_title("-W1/W2", fontsize=14)
        axes_w1w2.grid(True)

        # -W0/W2
        figure_w0w2 = Figure(figsize=(4.4, 2.3), dpi=55)
        figure_w0w2_canvas = FigureCanvasTkAgg(figure_w0w2)
        axes_w0w2 = figure_w0w2.add_subplot()
        figure_w0w2_canvas.get_tk_widget().place(x=380, y=457)
        axes_w0w2.set_title("-W0/W2", fontsize=14)
        axes_w0w2.grid(True)

        # Error
        figure_error_epoca = Figure(figsize=(6.7, 4), dpi=55)
        figure_error_epoca_canvas = FigureCanvasTkAgg(figure_error_epoca)
        axes_error_epoca = figure_error_epoca.add_subplot()
        figure_error_epoca_canvas.get_tk_widget().place(x=960, y=20)
        axes_error_epoca.set_title("Error de epoca", fontsize=16)
        axes_error_epoca.grid(True)

        # Comparison histogram
        figure_comparison = Figure(figsize=(6.7, 3.8), dpi=55)
        figure_comparison_canvas = FigureCanvasTkAgg(figure_comparison)
        axes_comparison = figure_comparison.add_subplot()
        figure_comparison_canvas.get_tk_widget().place(x=960, y=370)
        axes_comparison.set_title("Comparación", fontsize=16)
        axes_comparison.grid(True)



"""--------------------------------------------------------------------Initialization-------------------------------------------------------------------"""
# Variables for interface creation
main_window = tk.Tk()
selectMod = tk.IntVar()
breakTrain = False
app = Application(main_window)
app.mainloop()
