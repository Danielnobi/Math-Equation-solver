print("Loading...")

# common libraries
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
# import matplotlib.image as mpimg
# %matplotlib inline
import streamlit as st

# CV and Image
import cv2

# keras
import keras
from keras import optimizers
from keras import backend as K
from keras.models import Model, Sequential, load_model
K.image_data_format()

print("Done")

import numpy as np 
import pandas as pd 
# import matplotlib.pyplot as plt
import cv2
# from google.colab.patches import cv2_imshowh
from keras import backend as K

# sns.set()
from keras.models import model_from_json

github_link = "https://github.com/AryanGKulkarni/Math-Equation-solver"

# SVG code for the GitHub icon
github_svg = """
<svg xmlns="http://www.w3.org/2000/svg" height="30" width="30" viewBox="0 0 16 16" fill="currentColor">
    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
</svg>
"""

# HTML code to display GitHub icon with hyperlink
github_html = f'<a href="{github_link}" target="_blank">{github_svg}</a>'

# Display the GitHub icon and hyperlink
st.markdown(github_html, unsafe_allow_html=True)

st.title("Math Equation Solver")

# Load the model from the file
print("starting")
json_file = open('model_final.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(f"model_files/model_final.h5")
print("done")

img1='images/rada.png'
ww=0
img2 = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if img2 is None:
    st.write("Sample Image")
    ww=1
    img2=img1


if img2 is not None:
    if ww==0:
        st.image(img2, caption='Your Image', use_column_width=True)
        file_bytes = img2.getvalue()
        nparr = np.frombuffer(file_bytes, np.uint8)

        # Read the image using cv2.imdecode()
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        # plt.imshow(img)
    else:
        st.image(img2, caption='Your Image', use_column_width=True)
        img=cv2.imread(img2,cv2.IMREAD_GRAYSCALE)


    img=~img
    _,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ctrs,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cnt=sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    w=int(28)
    h=int(28)
    train_data=[]
    #print(len(cnt))
    rects=[]
    for c in cnt :
        x,y,w,h= cv2.boundingRect(c)
        rect=[x,y,w,h]
        rects.append(rect)
    bool_rect=[]
    for r in rects:
        l=[]
        for rec in rects:
            flag=0
            if rec!=r:
                if r[0]<(rec[0]+rec[2]+10) and rec[0]<(r[0]+r[2]+10) and r[1]<(rec[1]+rec[3]+10) and rec[1]<(r[1]+r[3]+10):
                    flag=1
                l.append(flag)
            if rec==r:
                l.append(0)
        bool_rect.append(l)
    dump_rect=[]
    for i in range(0,len(cnt)):
        for j in range(0,len(cnt)):
            if bool_rect[i][j]==1:
                area1=rects[i][2]*rects[i][3]
                area2=rects[j][2]*rects[j][3]
                if(area1==min(area1,area2)):
                    dump_rect.append(rects[i])
    #print(len(dump_rect)) 
    final_rect=[i for i in rects if i not in dump_rect]
    #print(final_rect)
    for r in final_rect:
        x=r[0]
        y=r[1]
        w=r[2]
        h=r[3]
        im_crop =thresh[y:y+h+10,x:x+w+10]
        im_resize = cv2.resize(im_crop,(28,28))
        im_resize=np.reshape(im_resize,(28,28,1))
        train_data.append(im_resize)
        

    for digit in train_data:
        prediction = model.predict(digit.reshape(1, 28, 28, 1))  
        
        #print ("\n\n---------------------------------------\n\n")
        #print ("=========PREDICTION============ \n\n")
        #plt.imshow(digit.reshape(28, 28), cmap="gray")
        #plt.show()
        #print("\n\nFinal Output: {}".format(np.argmax(prediction)))
        
        #print ("\nPrediction (Softmax) from the neural network:\n\n {}".format(prediction))
        
        hard_maxed_prediction = np.zeros(prediction.shape)
        hard_maxed_prediction[0][np.argmax(prediction)] = 1
        #print ("\n\nHard-maxed form of the prediction: \n\n {}".format(hard_maxed_prediction))
        #print ("\n\n---------------------------------------\n\n")

    equation=''

    for i in range(len(train_data)):
        
        train_data[i]=np.array(train_data[i])
        train_data[i]=train_data[i].reshape(1,28,28,1)
        result=np.argmax(model.predict(train_data[i]), axis=-1)
            
        for j in range(10) :
            if result[0] == j :
                equation = equation + str(j)
        
        if result[0] == 10 :
            equation = equation + "+"
        if result[0] == 11 :
            equation = equation + "-"
        if result[0] == 12 :
            equation = equation + "*"
        if result[0] == 13 :
            equation = equation + "/"
        if result[0] == 14 :
            equation = equation + "="
        if result[0] == 15 :
            equation = equation + "."
        if result[0] == 16 :
            equation = equation + "x"
        if result[0] == 17 :
            equation = equation + "y"      
        if result[0] == 18 :
            equation = equation + "z"

        s=equation
    print("Your Equation:", equation)
    t=""
    i=0
    while i<len(s):
        if s[i]=="-" and s[i+1]=="-":
            t=t+"="
            i=i+2
        elif s[i]=="=" and s[i+1]=="=":
            t=t+"="
            i=i+2
        else: 
            t=t+s[i]
            i=i+1

    equation=t
        
    #print("Your Equation :", equation)
    import sympy as sp
    def lsttostr(lst):
        str = ""
        for i in lst:
            str += i
        return str

    equation = list(equation)
    temp2 = 0
    for i in equation:
        if i == '=':
            equation[temp2] = '-('
            equation += ')'
        temp2 += 1
    equation = lsttostr(equation)

    alpha = 'abcdefghijklmnopqrstuvwxyz'
    #equation = list(equation)

    temp = 0

    for i in equation:
        if i in alpha:
            var = i
            #equation[equation.index(i)] = 'a'
            temp = 1
    #equation = lsttostr(equation)
    #print(equation)

    x = sp.symbols(var)
    #eq_raw = eval(equation)
    #print(eq_raw)
    if temp != 0:
        s=equation
        t=""
        i=0
        while i<len(s):       
            if (s[i]==var) and (s[i+1] in '123456789'):
                t=t+s[i]+"**"+s[i+1]
                i=i+2
            else: 
                t=t+s[i]
                i=i+1       
        equation=t    
        print("Your Equation:", equation)
        st.write("Your Equation: ", equation)
        s=equation
        t=""
        i=0
        while i<len(s):
            if (s[i] in '123456789') and (s[i+1]==var):
                t=t+s[i]+"*"+s[i+1]
                i=i+2        
            else: 
                t=t+s[i]
                i=i+1

        equation=t    
        eq_raw = eval(equation)    
        print(equation)
        eq = sp.Eq(eq_raw, 0)
        print("Solution:", sp.solve(eq_raw, x))
        st.write("Solution: ", sp.solve(eq_raw, x))
        #display(sp.solve(eq_raw, x))
    else:    
        eq_raw = eval(equation)
        print(equation)
        print("Solution:", eq_raw)
        st.write("Solution: ", eq_raw)
else:
    st.write("Please upload an image")