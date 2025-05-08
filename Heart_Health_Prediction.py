import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset and train model
df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# GUI App
main_window = tk.Tk()
main_window.title("Heart Disease Predictor")
main_window.geometry("847x487")

# Optional background image
try:
    bg_image = tk.PhotoImage(file="C:/Users/divij/source/repos/Heart_Health_Prediction/Heart_Health_Prediction/main_window_bg.png")
    bg_label = tk.Label(main_window, image=bg_image)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
except:
    print("Background image not found or failed to load.")

# Labels and Entry fields
age_label = tk.Label(main_window, text="Age", font=('Arial', 8), anchor='w')
age_label.place(x=40, y=210)
age_entry = tk.Entry(main_window, width=10)
age_entry.place(x=220, y=210)

sex_label = tk.Label(main_window, text="Sex (1=Male, 0=Female)", font=('Arial', 8), anchor='w')
sex_label.place(x=40, y=240)
sex_entry = tk.Entry(main_window, width=10)
sex_entry.place(x=220, y=240)

cp_label = tk.Label(main_window, text="Chest Pain Type (0-3)", font=('Arial', 8), anchor='w')
cp_label.place(x=40, y=270)
cp_entry = tk.Entry(main_window, width=10)
cp_entry.place(x=220, y=270)

restbp_label = tk.Label(main_window, text="Resting BP", font=('Arial', 8), anchor='w')
restbp_label.place(x=40, y=300)
restbp_entry = tk.Entry(main_window, width=10)
restbp_entry.place(x=220, y=300)

chol_label = tk.Label(main_window, text="Cholesterol", font=('Arial', 8), anchor='w')
chol_label.place(x=40, y=330)
chol_entry = tk.Entry(main_window, width=10)
chol_entry.place(x=220, y=330)

fbs_label = tk.Label(main_window, text="Fasting Blood Sugar\n(>120 mg/dl: 1, else: 0)", font=('Arial', 8), anchor='w')
fbs_label.place(x=40, y=360)
fbs_entry = tk.Entry(main_window, width=10)
fbs_entry.place(x=220, y=360)

restecg_label = tk.Label(main_window, text="RestECG (0-2)", font=('Arial', 8), anchor='w')
restecg_label.place(x=40, y=410)
restecg_entry = tk.Entry(main_window, width=10)
restecg_entry.place(x=220, y=410)

thalach_label = tk.Label(main_window, text="Max Heart Rate", font=('Arial', 8), anchor='w')
thalach_label.place(x=310, y=210)
thalach_entry = tk.Entry(main_window, width=10)
thalach_entry.place(x=440, y=210)

exang_label = tk.Label(main_window, text="Exercise Induced Angina\n (1=yes, 0=no)", font=('Arial', 8), anchor='w')
exang_label.place(x=310, y=340)
exang_entry = tk.Entry(main_window, width=10)
exang_entry.place(x=440, y=340)

oldpeak_label = tk.Label(main_window, text="Oldpeak", font=('Arial', 8), anchor='w')
oldpeak_label.place(x=310, y=270)
oldpeak_entry = tk.Entry(main_window, width=10)
oldpeak_entry.place(x=440, y=270)

slope_label = tk.Label(main_window, text="Slope (0-2)", font=('Arial', 8), anchor='w')
slope_label.place(x=310, y=300)
slope_entry = tk.Entry(main_window, width=10)
slope_entry.place(x=440, y=300)

ca_label = tk.Label(main_window, text="CA (0-4)", font=('Arial', 8), anchor='w')
ca_label.place(x=310, y=240)
ca_entry = tk.Entry(main_window, width=10)
ca_entry.place(x=440, y=240)

thal_label = tk.Label(main_window, text="Thal (1=normal;\n 2=fixed defect;\n 3=reversible defect)", font=('Arial', 8), anchor='w')
thal_label.place(x=310, y=390)
thal_entry = tk.Entry(main_window, width=10)
thal_entry.place(x=440, y=390)

name_label = tk.Label(main_window, text="NAME :", font=('Arial', 10), anchor='w')
name_label.place(x=600, y=290)
name_entry = tk.Entry(main_window, width=20 ,)
name_entry.place(x=670, y=290)



def predict():
    try:
        user_input = [
            float(age_entry.get()),
            float(sex_entry.get()),
            float(cp_entry.get()),
            float(restbp_entry.get()),
            float(chol_entry.get()),
            float(fbs_entry.get()),
            float(restecg_entry.get()),
            float(thalach_entry.get()),
            float(exang_entry.get()),
            float(oldpeak_entry.get()),
            float(slope_entry.get()),
            float(ca_entry.get()),
            float(thal_entry.get())
        ]
        user_name = name_entry.get().upper()
        print(user_name)

        input_scaled = scaler.transform([user_input])
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]


        if prediction == 1:
            result = f"\u26a0\ufe0f High Risk of Heart Disease "
        else:
            result = f"\u2705 Low Risk of Heart Disease "

        result_window = tk.Toplevel(main_window)
        result_window.title("Heart Disease Predictor")
        result_window.geometry("847x487")

# Optional background image
        try:
            bg_image_result = tk.PhotoImage(file="C:/Users/divij/source/repos/Heart_Health_Prediction/Heart_Health_Prediction/result_window_bg.png")
            bg_label_result = tk.Label(result_window, image=bg_image_result)
            bg_label_result.place(x=0, y=0, relwidth=1, relheight=1)
            bg_label_result.image = bg_image_result

            name_label = tk.Label(result_window, text=f"Name        : {user_name}", font=('Arial', 15), )
            name_label.place(x=50, y=215)
            name_label.config(bg="white")

            pred_label = tk.Label(result_window, text=f"HEALTH      : {result}", font=('Arial', 15), )
            pred_label.place(x=50, y=285)
            pred_label.config(bg="white")

            prob_label = tk.Label(result_window, text=f"PROBABILITY : {prob*100:.1f}%", font=('Arial', 15), )
            prob_label.place(x=50, y=355)
            prob_label.config(bg="white")

            button2 =tk.Button(
                result_window, text="Compare With Average", command=lambda: plot_user_input_vs_average(user_input),
                bg="red", fg="white", font=('Arial', 11), padx=10, pady=5
              )
            button2.place(x=600, y=300)


        except:
            print("Background image not found or failed to load.")

    except Exception as e:
         messagebox.showerror("Input Error", f"Please enter valid numbers.\n\n{str(e)}")
 
def plot_user_input_vs_average(user_input):
    avg_vals = df.drop("target", axis=1).mean().values
    features = df.columns[:-1]

    x = np.arange(len(features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, avg_vals, width, label='Average')
    ax.bar(x + width/2, user_input, width, label='Your Input')

    ax.set_ylabel('Values')
    ax.set_title('Comparison of Your Inputs vs Average')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=90)
    ax.legend()

    plt.tight_layout()
    plt.show()

button =tk.Button(
    main_window, text="Predict Heart Health", command=predict,
    bg="green", fg="white", font=('Arial', 11), padx=10, pady=5
)
button.place(x=620, y=340)


main_window.mainloop()