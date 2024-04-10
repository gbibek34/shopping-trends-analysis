import tkinter as tk
from tkinter import ttk
import pandas as pd
from knn_model import ShoppingBehaviorModel

class ShoppingBehaviorApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Shopping Behavior Classification")
        self.geometry("1055x500+200+100")

        self.trained = False
        #=================Style=========================
        self.style = ttk.Style()
        self.style.configure('My.TFrame', background='#ECECEC')
        self.style.configure('My.TLabel', background='#ECECEC', font=('Inter', 12))
        self.style.configure('My.TButton', foreground='blue', background='lightgray', font=("Inter", 12, "bold"), relief=tk.RAISED)
        self.style.configure('My.TRadiobutton', background='#ECECEC', font=('Inter', 12))
        
        #=================Frames=========================
        self.display_frame = ttk.Frame(self, style='My.TFrame', relief=tk.GROOVE, borderwidth=5)
        self.display_frame.place(x=10, y=10, width=430, height=485)

        self.option_frame = ttk.Frame(self.display_frame, style='My.TFrame', relief=tk.GROOVE, borderwidth=5)
        self.option_frame.place(x=0, y=425, width=420, height=50)

        self.progress_frame = ttk.Frame(self.display_frame, style='My.TFrame', relief=tk.GROOVE, borderwidth=5)
        self.progress_frame.place(x=0, y=375, width=420, height=50)

        self.table_frame = ttk.Frame(self, style='My.TFrame', relief=tk.GROOVE, borderwidth=5)
        self.table_frame.place(x=450, y=10, width=600, height=485)

        self.tree_frame = ttk.Frame(self.table_frame, style='My.TFrame', relief=tk.GROOVE, borderwidth=5)
        self.tree_frame.place(x=0, y=55, height=420, width=590)

        self.title_frame = ttk.Frame(self.table_frame, relief=tk.GROOVE, borderwidth=5)
        self.title_frame.place(x=0, y=0, width=590, height=50)
        #=================Labels=========================
        self.label_age = ttk.Label(self.display_frame, text="Age:", style='My.TLabel')
        self.label_age.grid(row=0, column=0, padx=10, pady=5, sticky='w')

        self.label_gender = ttk.Label(self.display_frame, text="Gender:", style='My.TLabel')
        self.label_gender.grid(row=1, column=0, padx=10, pady=5, sticky='w')

        self.label_location = ttk.Label(self.display_frame, text="Location:", style='My.TLabel')
        self.label_location.grid(row=2, column=0, padx=10, pady=5, sticky='w')

        self.title_label = ttk.Label(self.title_frame, text="Used columns from the Training Data", font=("Inter", 15, "bold"))
        self.title_label.pack(fill=tk.BOTH,expand=True)

        self.result_label = ttk.Label(self.display_frame, text="", style='My.TLabel')
        self.result_label.grid(row=4, column=0, columnspan=3, padx=10, pady=5, sticky='w')

        #=================Entries=========================
        self.entry_age = ttk.Entry(self.display_frame)
        self.entry_age.grid(row=0, column=1, padx=10, pady=5, sticky='ew')
        self.entry_age.config(validate="key", validatecommand=(self.register(self.validate_age), "%P"))

        self.gender_var = tk.StringVar()
        self.gender_male_radiobutton = ttk.Radiobutton(self.display_frame, text="Male", value="Male", variable=self.gender_var, style='My.TRadiobutton')
        self.gender_male_radiobutton.grid(row=1, column=1, padx=10, pady=5, sticky='w')
        self.gender_female_radiobutton = ttk.Radiobutton(self.display_frame, text="Female", value="Female", variable=self.gender_var, style='My.TRadiobutton')
        self.gender_female_radiobutton.grid(row=1, column=2, padx=10, pady=5, sticky='w')

        self.city_var = tk.StringVar()
        self.city_combobox = ttk.Combobox(self.display_frame, textvariable=self.city_var, state="readonly")
        self.city_combobox.grid(row=2, column=1, columnspan=2, padx=10, pady=5, sticky='ew')
        #=================Buttons=========================
        if self.trained == False:
            self.train_btn = ttk.Button(self.option_frame, text="Train Model", style='My.TButton', command=self.start_progress)
            self.train_btn.pack(fill=tk.BOTH, expand=True)
        else:
            self.predict_btn = ttk.Button(self.option_frame, text="Predict", style='My.TButton', command=self.predict)
            self.predict_btn.pack(fill=tk.BOTH, expand=True)

        #=================Progress Bar=========================
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", length=200, mode="indeterminate")
        self.progress_bar.pack(fill=tk.BOTH, expand=True)

        #=================Tree View=========================
        self.scroll_y = ttk.Scrollbar(self.tree_frame, orient=tk.VERTICAL)
        self.data_tbl = ttk.Treeview(self.tree_frame,
                                    columns=("Age", "Gender", "Location", "Item Purchased"),
                                    xscrollcommand=self.scroll_y.set)
        self.scroll_y.pack(side=tk.RIGHT, fill='y')
        self.scroll_y.config(command=self.data_tbl.yview)
        self.data_tbl.heading("Age",text="Age")
        self.data_tbl.heading("Gender",text="Gender")
        self.data_tbl.heading("Location",text="Location")
        self.data_tbl.heading("Item Purchased", text="Item Purchased")
        self.data_tbl['show']='headings'
        self.data_tbl.column("Age", width=15)
        self.data_tbl.column("Gender", width=20)
        self.data_tbl.column("Location", width=20)
        self.data_tbl.column("Item Purchased", width=50)
        self.data_tbl.pack(fill=tk.BOTH, expand='1')
        self.insert_data()

    def insert_data(self):
        self.data = pd.read_csv('data/preprocessed_data.csv')
        for index, row in self.data.iterrows():
            self.data_tbl.insert("", "end", values=(row['Age'], row['Gender'], row['Location'], row['Item Purchased']))

    def start_progress(self):
        self.progress_bar.start()
        self.model = ShoppingBehaviorModel('data\preprocessed_data.csv')
        self.after(5000, self.load_cities)

    def load_cities(self):
        cities = self.model.df['Location'].unique().tolist()
        self.city_combobox['values'] = cities
        self.trained = True
        if self.trained:
            self.train_btn.pack_forget()
            self.predict_btn = ttk.Button(self.option_frame, text="Predict", style='My.TButton', command=self.predict)
            self.predict_btn.pack(fill=tk.BOTH, expand=True)
        self.progress_bar.stop()
        self.update()

    def validate_age(self, new_value):
        if new_value.isdigit():
            return True
        elif new_value == "":
            return True
        else:
            return False
    
    def validate_inputs(self):
        age = self.entry_age.get().strip()
        gender = self.gender_var.get()
        location = self.city_var.get().strip()

        if age == "" or gender == "" or location == "":
            return False
        return True

    def predict(self):
        if not self.validate_inputs():
            self.result_label.config(text="Please fill in all fields.")
            return

        user_age = int(self.entry_age.get())
        user_gender = self.gender_var.get()
        user_location = self.city_var.get()

        predicted_item, predicted_payment_decoded = self.model.predict(user_age, user_gender, user_location)
        
        # Join the elements of the predicted item and payment lists into strings
        predicted_item_string = ',\n'.join(predicted_item)
        predicted_payment_string = ', '.join(predicted_payment_decoded)

        # Set the result label text with the joined strings
        self.result_label.config(text=f"Predicted item:\n{predicted_item_string}\n\nPredicted payment method: {predicted_payment_string}")

if __name__ == "__main__":
    app = ShoppingBehaviorApp()
    app.mainloop()
