import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass


def classify_features(selected_features):
    preparation = ['Pyro_Temp', 'Hold_D', 'Heat_R', 'Acid_Type', 'Acid_C',
                   'Acid_T', 'Acid_Temp', 'Seq_P_M']
    physicochemical = ['SSA', 'O/C', 'pHpzc']
    adsorption = ['pH', 'T', 'C0', 'SLR']

    return {
        "Preparation Conditions": preparation,
        "Physicochemical Properties": physicochemical,
        "Adsorption Conditions": adsorption
    }


def load_model_and_features():
    try:
        df = pd.read_excel('merge_dataset_MICE.xlsx')
        target = 'Qe'
        selected_features = ['Pyro_Temp', 'Hold_D', 'Heat_R', 'Acid_Type', 'Acid_C',
                             'Acid_T', 'Acid_Temp', 'Seq_P_M', 'SSA', 'O/C',
                             'pHpzc', 'pH', 'T', 'C0', 'SLR']

        X = df[selected_features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )

        model_low = CatBoostRegressor(
            loss_function='Quantile:alpha=0.025',
            random_state=42,
            verbose=False
        )
        model_low.fit(X_train, y_train)

        model_high = CatBoostRegressor(
            loss_function='Quantile:alpha=0.975',
            random_state=42,
            verbose=False
        )
        model_high.fit(X_train, y_train)

        model_median = CatBoostRegressor(
            loss_function='Quantile:alpha=0.5',
            random_state=42,
            verbose=False
        )
        model_median.fit(X_train, y_train)

        y_test_pred = model_median.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)

        feature_groups = classify_features(selected_features)

        return {
            'model_low': model_low,
            'model_high': model_high,
            'model_median': model_median,
            'feature_groups': feature_groups,
            'all_features': selected_features,
            'test_rmse': round(test_rmse, 2),
            'test_r2': round(test_r2, 4)
        }
    except FileNotFoundError as e:
        messagebox.showerror("File Error", f"Model/feature file not found:\n{e}")
        raise SystemExit()
    except Exception as e:
        messagebox.showerror("Load Error", f"Failed to load model/data:\n{e}")
        raise SystemExit()


class QePredictGUI:
    def __init__(self, root, model_info):
        self.root = root
        self.model_info = model_info
        self.input_entries = {}
        self.root.tk.call('tk', 'scaling', 2.0)

        self.feature_units = {
            'Pyro_Temp': '(°C)',
            'Hold_D': '(h)',
            'Heat_R': '(°C/min)',
            'Acid_C': '(mol/L)',
            'Acid_T': '(h)',
            'Acid_Temp': '(°C)',
            'Seq_P_M': '',
            'Acid_Type': '',
            'SSA': '(m²/g)',
            'O/C': '(%)',
            'pHpzc': '',
            'pH': '',
            'T': '(K)',
            'C0': '(mg/L)',
            'SLR': '(g/L)'
        }

        self._setup_style()
        self._setup_layout()

    def _setup_style(self):
        self.style = ttk.Style(self.root)

        self.colors = {
            'title_bg': '#ABC6E4',
            'title_text': '#000000',
            'section_bg': '#D9E5F3',
            'section_title_bg': '#FFFACD',
            'section_border': '#4a86e8',
            'input_bg': '#ffffff',
            'button_bg': '#ABC6E4',
            'button_hover': '#80b8ff',
            'result_bg': '#ABC6E4',
            'text_normal': '#000000',
            'text_highlight': '#000000'
        }

        self.fonts = {
            'title': ('Times New Roman', 20, 'bold'),
            'section': ('Times New Roman', 14, 'bold'),
            'label': ('Times New Roman', 12, 'bold'),
            'entry': ('Times New Roman', 12),
            'button': ('Times New Roman', 14, 'bold'),
            'result': ('Times New Roman', 16, 'bold'),
            'unit': ('Times New Roman', 12, 'bold')
        }

        self.style.configure('Title.TFrame', background=self.colors['title_bg'])
        self.style.configure('Title.TLabel',
                             background=self.colors['title_bg'],
                             foreground=self.colors['title_text'],
                             font=self.fonts['title'])

        self.style.configure('Section.TLabelframe',
                             background=self.colors['section_bg'],
                             bordercolor=self.colors['section_border'],
                             relief='solid',
                             borderwidth=3)
        self.style.configure('Section.TLabelframe.Label',
                             background=self.colors['section_title_bg'],
                             foreground=self.colors['text_normal'],
                             font=self.fonts['section'],
                             padding=(10, 5))

        self.style.configure('Input.TEntry',
                             fieldbackground=self.colors['input_bg'],
                             font=self.fonts['entry'],
                             padding=8,
                             borderwidth=2,
                             relief='solid')

        self.style.configure('Predict.TButton',
                             font=self.fonts['button'],
                             padding=(15, 10),
                             width=25,
                             background=self.colors['button_bg'],
                             foreground='#000000',
                             borderwidth=2,
                             relief='raised')
        self.style.map('Predict.TButton',
                       background=[('active', self.colors['button_hover']),
                                   ('pressed', '#2a66c8'),
                                   ('!active', self.colors['button_bg'])])

        self.style.configure('Result.TFrame',
                             background=self.colors['result_bg'])
        self.style.configure('Result.TLabel',
                             font=self.fonts['result'],
                             background=self.colors['result_bg'],
                             foreground='#000000',
                             padding=10)

        self.style.configure('Inner.TFrame',
                             background=self.colors['section_bg'])

        self.style.configure('Unit.TLabel',
                             font=self.fonts['unit'],
                             background=self.colors['section_bg'],
                             foreground=self.colors['text_normal'])

    def _setup_layout(self):
        self.root.title("Biochar Adsorption Capacity (Qe) Predictor")
        self.root.geometry("1600x1000")
        self.root.resizable(True, True)

        title_frame = ttk.Frame(self.root, style='Title.TFrame', height=80)
        title_frame.pack(fill=tk.X, padx=0, pady=0)
        title_frame.pack_propagate(False)

        title_label = ttk.Label(
            title_frame,
            text="Biochar Adsorption Capacity (Qe) Predictor",
            style='Title.TLabel'
        )
        title_label.pack(expand=True)

        input_frame = ttk.Frame(self.root, padding=20)
        input_frame.pack(fill=tk.BOTH, expand=True)

        col1 = ttk.Frame(input_frame)
        col2 = ttk.Frame(input_frame)
        col3 = ttk.Frame(input_frame)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        prep_frame = ttk.LabelFrame(
            col1,
            text="Preparation Conditions",
            style='Section.TLabelframe'
        )
        prep_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self._add_feature_entries(prep_frame,
                                  self.model_info['feature_groups']["Preparation Conditions"])

        phys_frame = ttk.LabelFrame(
            col2,
            text="Physicochemical Properties",
            style='Section.TLabelframe'
        )
        phys_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self._add_feature_entries(phys_frame,
                                  self.model_info['feature_groups']["Physicochemical Properties"])

        adsorb_frame = ttk.LabelFrame(
            col3,
            text="Adsorption Conditions",
            style='Section.TLabelframe'
        )
        adsorb_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self._add_feature_entries(adsorb_frame,
                                  self.model_info['feature_groups']["Adsorption Conditions"])

        result_frame = ttk.Frame(self.root, style='Result.TFrame', height=120)
        result_frame.pack(fill=tk.X, padx=0, pady=0)
        result_frame.pack_propagate(False)

        center_frame = ttk.Frame(result_frame, style='Result.TFrame')
        center_frame.pack(expand=True, fill=tk.BOTH)

        median_label = ttk.Label(
            center_frame,
            text="Qe Prediction Value:",
            style='Result.TLabel'
        )
        median_label.pack(side=tk.LEFT, padx=(15, 5), pady=15)

        self.median_var = tk.StringVar(value="")
        median_entry = ttk.Entry(
            center_frame,
            textvariable=self.median_var,
            style='Input.TEntry',
            width=15,
            font=self.fonts['result'],
            state='readonly',
            justify='center'
        )
        median_entry.pack(side=tk.LEFT, padx=5, pady=15)

        ttk.Label(
            center_frame,
            text="mg/g",
            style='Result.TLabel'
        ).pack(side=tk.LEFT, padx=(0, 30), pady=15)

        interval_label = ttk.Label(
            center_frame,
            text="95% Prediction Interval:",
            style='Result.TLabel'
        )
        interval_label.pack(side=tk.LEFT, padx=(15, 5), pady=15)

        self.interval_var = tk.StringVar(value="")
        interval_entry = ttk.Entry(
            center_frame,
            textvariable=self.interval_var,
            style='Input.TEntry',
            width=25,
            font=self.fonts['result'],
            state='readonly',
            justify='center'
        )
        interval_entry.pack(side=tk.LEFT, padx=5, pady=15)

        ttk.Label(
            center_frame,
            text="mg/g",
            style='Result.TLabel'
        ).pack(side=tk.LEFT, padx=(0, 15), pady=15)

        button_frame = ttk.Frame(self.root, padding=20)
        button_frame.pack(fill=tk.X, pady=15)

        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        predict_btn = ttk.Button(
            button_frame,
            text="Predict",
            command=self._predict_qe,
            style='Predict.TButton'
        )
        predict_btn.grid(row=0, column=0, padx=15, sticky="ew")

        reset_btn = ttk.Button(
            button_frame,
            text="Reset",
            command=self._reset_inputs,
            style='Predict.TButton'
        )
        reset_btn.grid(row=0, column=1, padx=15, sticky="ew")

    def _add_feature_entries(self, parent_frame, features):
        inner_frame = ttk.Frame(parent_frame, style='Inner.TFrame')
        inner_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        for idx, feature in enumerate(features, 1):
            row_frame = ttk.Frame(inner_frame, style='Inner.TFrame')
            row_frame.pack(fill=tk.X, pady=12)

            label = ttk.Label(
                row_frame,
                text=f"{feature}",
                font=self.fonts['label'],
                foreground=self.colors['text_highlight'],
                background=self.colors['section_bg']
            )
            label.pack(side=tk.LEFT, padx=8, anchor=tk.W)

            entry = ttk.Entry(
                row_frame,
                style='Input.TEntry',
                width=18,
                justify='right'
            )
            entry.pack(side=tk.RIGHT, padx=(8, 4))

            if feature in self.feature_units:
                unit_label = ttk.Label(
                    row_frame,
                    text=self.feature_units[feature],
                    style='Unit.TLabel'
                )
                unit_label.pack(side=tk.RIGHT, padx=(4, 8))

            self.input_entries[feature] = entry

    def _get_valid_inputs(self):
        input_data = {}
        for feature, entry in self.input_entries.items():
            input_val = entry.get().strip()
            if not input_val:
                messagebox.showwarning("Input Warning", f"Please enter value for '{feature}'")
                return None
            try:
                if feature == 'Acid_Type':
                    input_data[feature] = [input_val]
                else:
                    input_data[feature] = [float(input_val)]
            except ValueError:
                messagebox.showerror("Input Error", f"'{feature}' must be a number")
                return None

        input_df = pd.DataFrame(input_data)
        input_df = input_df.reindex(columns=self.model_info['all_features'], fill_value=0)
        return input_df

    def _predict_qe(self):
        input_df = self._get_valid_inputs()
        if input_df is None:
            return

        try:
            pred_median = self.model_info['model_median'].predict(input_df)[0]
            pred_low = self.model_info['model_low'].predict(input_df)[0]
            pred_high = self.model_info['model_high'].predict(input_df)[0]

            if pred_low > pred_high:
                pred_low, pred_high = pred_high, pred_low

            self.median_var.set(f"{pred_median:.2f}")
            self.interval_var.set(f"[{pred_low:.2f}, {pred_high:.2f}]")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Failed to predict: {str(e)}")

    def _reset_inputs(self):
        for entry in self.input_entries.values():
            entry.delete(0, tk.END)
        self.median_var.set("")
        self.interval_var.set("")


if __name__ == "__main__":
    model_info = load_model_and_features()
    root = ThemedTk(theme="arc")
    app = QePredictGUI(root, model_info)
    root.mainloop()