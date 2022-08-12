
## MATH EQUATION SOLVER

This project is a simple machine learning model which recognizes 
handwritten equations from an image and solve it.

We are given the problem to make a program to recognize a handwritten equation solver using Machine Learning. We have made a model which can recognize a handwritten equation from an image and then solve it. What kind of equation? Our model can recognize and solve basic and not-so-basic numerical expressions like 2+4 and e5 + 7, equations of one variable up to degree 2. This model works on the CNN architecture and has been made using various python libraries like TensorFlow, sympy, OpenCV etc



The working of the model is divided into two parts:

•	Recognition of the Equation

•	Solving the Equation

Recognition of Equation:

After the image input is fed into the model, it identifies the digits and draws contours around them. After that, it recognizes the digit or math symbols present in that contour.

Solving the Equation:

After the Recognition is done the system checks for a variable in the equation. If it recognizes a variable in the equation, it solves the equation for that variable, otherwise, it interprets the equation as a numeric expression and solves it.






## Working

There are two files; model.ipynb and main.ipynb containing the
model and equation solver respectively.

To run the program open main.ipynb and first write the image path containing the 
equation in the following code:

```
img = cv2.imread('write your path here',cv2.IMREAD_GRAYSCALE)

```
Now run the program to get the final output.
## Main.ipynb

This file contains the main program of our equation solver including the trained model  
## Team Members

[Aryan Kulkarni](https://github.com/AryanGKulkarni)

[Harshit Kumar Singh](https://github.com/harshit-ji)

[KSS Abhinav](https://github.com/abhinav180104)

[Dhanush Reddy](https://github.com/dhanushreddy2)




## Demo



[Demo Video](https://drive.google.com/file/d/16AQa5amE3IE6HrFKpVT8poIpUiF6QuDf/view?usp=sharing)
## Mentors
[Priyansh Jaseja](https://github.com/iDroppiN)

[Snehit Chinta](https://github.com/snehithchinta)
## Model.ipynb
This file contains the definition of the model, training of the model and testing of the model.