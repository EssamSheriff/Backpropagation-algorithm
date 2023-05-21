from pathlib import Path
from tkinter import *
from tkinter import Tk

from main import *

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("./assets")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("1284x728")
window.configure(bg="#054F54")

canvas = Canvas(
    window,
    bg="#054F54",
    height=728,
    width=1284,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)

canvas.place(x=0, y=0)
button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))


# ######################################################################
# here the functions
def buClicked():
    Biasflag = 0
    if var1_flag.get() == 'Yes':
        Biasflag = 1


    if f1_v1.get() == 'Yes':
        ActivationFun = 1
    elif f2_v1.get() == 'Yes':
        ActivationFun = 0
    else:
        ActivationFun = -1


    ep = Epochs_entry.get()
    if ep == '':
        ep = int('-2')
    else:
        try:
            ep = int(ep)
        except ValueError:
            ep = int('-2')

    eta = entry_2.get()
    if eta == '':
        eta = int('-2')
    else:
        try:
            eta = float(eta)
        except ValueError:
            eta = float('-2')

    NumHiddenLayers = Hidden_Layers_entry.get()
    if NumHiddenLayers == '':
        NumHiddenLayers = 1
    else:
        try:
            NumHiddenLayers = int(NumHiddenLayers)
        except ValueError:
            NumHiddenLayers = 1

    NumNeuronsInHiddenLayers_str = Neurons_in_Hidden_Layers_entry.get()
    if NumNeuronsInHiddenLayers_str == '':
        NumNeuronsInHiddenLayers = [1]
    else:
        try:
            NumNeuronsInHiddenLayers = NumNeuronsInHiddenLayers_str.split(",")
        except ValueError:
            NumNeuronsInHiddenLayers = [1]

    print("Epochs= ", ep, ",Learning Rate= ", eta, ",Biasflag= ",Biasflag, ",Activate= ",ActivationFun,
           "\n,NumHiddenLayers= ", NumHiddenLayers, ",NeuronsInHiddenLayers= ", NumNeuronsInHiddenLayers)

    if ep > -1 and eta > -1 and ActivationFun != -1 :
        test_acc, train_acc = run(NumHiddenLayers, NumNeuronsInHiddenLayers, ActivationFun, eta, ep, Biasflag)

        Test_Acc_Str.set(str(round(test_acc, 2)) + " %")
        Train_Acc_Str.set(str(round(train_acc, 2)) +" %")

def FeaturesValidation():
    i = 0
    if f1_v1.get() == 'Yes': i = i + 1
    if f2_v1.get() == 'Yes': i = i + 1
    if (i >= 1):
        if f1_v1.get() != 'Yes': f1.config(state="disabled")
        if f2_v1.get() != 'Yes': f2.config(state="disabled")
    else:
        f1.config(state="normal")
        f2.config(state="normal")


# ######################################################################
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: buClicked(),
    relief="flat"
)

button_1.place(
    x=74.0,
    y=590.9459991455078,
    width=538.0,
    height=94.0
)

entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    1009.0,
    364.44598388671875,
    image=entry_image_1
)
Epochs_entry = Entry(
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0
)
Epochs_entry.place(
    x=829.0,
    y=342.94598388671875,
    width=360.0,
    height=41.0
)

entry_image_2 = PhotoImage(
    file=relative_to_assets("entry_2.png"))
entry_bg_2 = canvas.create_image(
    1009.0,
    233.44598388671875,
    image=entry_image_2
)
entry_2 = Entry(
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0
)
entry_2.place(
    x=829.0,
    y=211.94598388671875,
    width=360.0,
    height=41.0
)

Test_Acc_Str = StringVar()
Test_Acc_entry = Entry(
    bd=0,
    bg="#FFFFFF",
    justify=CENTER,
    state=DISABLED,
    textvariable=Test_Acc_Str,
    highlightthickness=0
)
Test_Acc_entry.place(
    x=750.0,
    y=620,
    width=100.0,
    height=50.0
)

Train_Acc_Str = StringVar()
Train_Acc_entry = Entry(
    bd=0,
    bg="#FFFFFF",
    justify=CENTER,
    state=DISABLED,
    textvariable=Train_Acc_Str,
    highlightthickness=0
)

Train_Acc_entry.place(
    x=1000,
    y=620,
    width=100.0,
    height=50.0
)

Hidden_Layers_entry = Entry(
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0
)

Hidden_Layers_entry.place(
    x=100,
    y=211.94598388671875,
    width=300,
    height=40
)

Neurons_in_Hidden_Layers_entry = Entry(
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0
)

Neurons_in_Hidden_Layers_entry.place(
    x=100,
    y=342.94598388671875,
    width=300,
    height=40
)
# ##################################### features #########################################
# hna lw howa ekhtar ay feature variable.get() htb2a 'Yes' w lw l2 htb2a 'No'
# f1_v1.get() = 'Yes'
f1_v1 = StringVar(window)
f1_v1.set('No')
f1 = Checkbutton(window, text='Sigmoid ', variable=f1_v1, onvalue='Yes', offvalue='No', command=FeaturesValidation)
f1.grid(row=1, column=1)
f2_v1 = StringVar(window)
f2_v1.set('No')
f2 = Checkbutton(window, text='Hyperbolic Tangent sigmoid ', variable=f2_v1, onvalue='Yes', offvalue='No', command=FeaturesValidation)
f2.grid(row=1, column=2)


f1.place(
    x=82.0,
    y=450.0,
    width=140.0,
    height=50.0
)

f2.place(
    x=182.0,
    y=450.0,
    width=250.0,
    height=50.0
)

########################################################################################


# ##################################### classes #########################################


######################################################################################
var1_flag = StringVar(window)
var1_flag.set('No')
var1 = Checkbutton(window, text="bias", variable=var1_flag, onvalue='Yes', offvalue='No')
var1.place(
    x=829.0,
    y=450.0,
    width=344.0,
    height=46.0
)

canvas.create_text(
    832.0000305175781,
    163.94598388671875,
    anchor="nw",
    text="Learning Rate",
    fill="#FFFFFF",
    font=("Inter", 32 * -1)
)

canvas.create_text(
    82.0,
    163.94598388671875,
    anchor="nw",
    text="# Hidden Layers",
    fill="#FFFFFF",
    font=("Inter", 32 * -1)
)
canvas.create_text(
    82.0,
    294.94598388671875,
    anchor="nw",
    text="# Neurons in each hidden layer",
    fill="#FFFFFF",
    font=("Inter", 32 * -1)
)

canvas.create_text(
    82.0,
    400,
    anchor="nw",
    text="Activation Function",
    fill="#FFFFFF",
    font=("Inter", 32 * -1)
)

canvas.create_text(
    829.0000152587891,
    294.94598388671875,
    anchor="nw",
    text="Epochs",
    fill="#FFFFFF",
    font=("Inter", 32 * -1)
)

canvas.create_text(
    400,
    20,
    anchor="nw",
    text="Penguins Comparison",
    fill="#FAEBEB",
    font=("Italiana Regular", 48 * -1)
)

canvas.create_text(
    470,
    80,
    anchor="nw",
    text="CS_H5 Team",
    fill="#FAEBEB",
    font=("Italiana Regular", 48 * -1)
)

canvas.create_text(
    750.0000305175781,
    570,
    anchor="nw",
    text="Test Accuracy",
    fill="#FFFFFF",
    font=("Inter", 30 * -1)
)

canvas.create_text(
    1000,
    570,
    anchor="nw",
    text="Train Accuracy",
    fill="#FFFFFF",
    font=("Inter", 30 * -1)
)
'''
canvas.create_text(
    950,
    480,
    anchor="nw",
    text="confusion matrix",
    fill="#FFFFFF",
    font=("Inter", 42 * -1)
)

canvas.create_text(
    950,
    600,
    anchor="nw",
    text="C1",
    fill="#FFFFFF",
    font=("Inter", 30 * -1)
)
canvas.create_text(
    950,
    670,
    anchor="nw",
    text="C2",
    fill="#FFFFFF",
    font=("Inter", 30 * -1)
)
canvas.create_text(
    1050,
    550,
    anchor="nw",
    text="C1",
    fill="#FFFFFF",
    font=("Inter", 30 * -1)
)
canvas.create_text(
    1170,
    550,
    anchor="nw",
    text="C2",
    fill="#FFFFFF",
    font=("Inter", 30 * -1)
)
'''
window.resizable(False, False)
window.mainloop()
