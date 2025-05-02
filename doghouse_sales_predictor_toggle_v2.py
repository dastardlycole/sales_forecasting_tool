import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import os
import sys
import webbrowser
import pandas as pd

if getattr(sys, 'frozen', False):
    import io
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()


class ForecastAppToggle(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Doghouse Sales Forecasting Tool with Version Toggle")
        self.geometry("750x550")
        
        # Track CSV validation status
        self.csv_validated = False

        # ── Help button in title area ──────────────────────
        help_frame = tk.Frame(self)
        help_frame.pack(fill="x", padx=10, pady=5)
        
        help_btn = tk.Button(
            help_frame,
            text="How to Use",
            command=self.show_instructions,
            bg="#17a2b8",
            fg="white"
        )
        help_btn.pack(side="right")

        # ── Version toggle ──────────────────────────────
        version_frame = tk.Frame(self)
        version_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(version_frame, text="Version:").pack(side="left")
        
        self.version_var = tk.StringVar(value="dissertation")
        
        version_styles = ttk.Style()
        version_styles.configure("TRadiobutton", padding=3)
        
        diss_radio = ttk.Radiobutton(
            version_frame, 
            text="Dissertation (Extended Features)",
            variable=self.version_var,
            value="dissertation",
            style="TRadiobutton",
            command=self.reset_validation
        )
        diss_radio.pack(side="left", padx=10)
        
        company_radio = ttk.Radiobutton(
            version_frame, 
            text="Company (Standard)",
            variable=self.version_var,
            value="company",
            style="TRadiobutton",
            command=self.reset_validation
        )
        company_radio.pack(side="left", padx=10)

        # ── File chooser ───────────────────────────────
        frm = tk.Frame(self)
        frm.pack(fill="x", padx=10, pady=5)
        tk.Label(frm, text="Sales CSV:").pack(side="left")
        self.file_var = tk.StringVar()
        self.file_var.trace_add("write", self.reset_validation)  # Reset validation when file changes
        tk.Entry(frm, textvariable=self.file_var, width=40).pack(side="left", padx=5)
        tk.Button(frm, text="Browse", command=self.browse).pack(side="left")
        self.validate_btn = tk.Button(
            frm, 
            text="Validate", 
            command=self.validate_csv, 
            bg="#28a745", 
            fg="white"
        )
        self.validate_btn.pack(side="left", padx=5)

        # ── Version description ────────────────────────
        desc_frame = tk.Frame(self)
        desc_frame.pack(fill="x", padx=10, pady=5)
        
        self.desc_text = tk.StringVar()
        self.update_description()
        
        desc_label = tk.Label(
            desc_frame, 
            textvariable=self.desc_text,
            justify="left",
            wraplength=730,
            relief="groove",
            padx=10,
            pady=10,
            bg="#f0f0f0"
        )
        desc_label.pack(fill="x")
        
        # Track version changes to update description
        self.version_var.trace_add("write", self.update_description)

        # ── CSV Requirements section ────────────────────
        req_frame = tk.Frame(self)
        req_frame.pack(fill="x", padx=10, pady=5)
        
        self.req_text = tk.StringVar()
        self.update_requirements()
        
        req_label = tk.Label(
            req_frame, 
            textvariable=self.req_text,
            justify="left",
            wraplength=730,
            relief="groove",
            padx=10,
            pady=10,
            bg="#fff8dc"  # Light yellow background
        )
        req_label.pack(fill="x")
        
        # Update requirements when version changes
        self.version_var.trace_add("write", self.update_requirements)

        # ── Run button ────────────────────────────────
        run_frame = tk.Frame(self)
        run_frame.pack(fill="x", padx=10, pady=5)
        
        self.run_btn = tk.Button(
            run_frame, 
            text="Run Forecast", 
            command=self.run,
            bg="#007bff",
            fg="white",
            padx=20,
            pady=5,
            state="disabled"  # Disabled until validation
        )
        self.run_btn.pack()

        # ── Output console ────────────────────────────
        self.log = scrolledtext.ScrolledText(self, state="disabled", height=10)
        self.log.pack(fill="both", expand=True, padx=10, pady=5)
        
        # ── Status bar ─────────────────────────────────
        status_frame = tk.Frame(self)
        status_frame.pack(fill="x", side="bottom")
        
        self.status_text = tk.StringVar(value="Ready - Please validate CSV before running")
        status_label = tk.Label(
            status_frame, 
            textvariable=self.status_text,
            bd=1,
            relief="sunken",
            anchor="w",
            padx=5
        )
        status_label.pack(fill="x")

    def reset_validation(self, *args):
        """Reset validation state when file or version changes"""
        self.csv_validated = False
        self.run_btn.config(state="disabled")
        self.status_text.set("Ready - Please validate CSV before running")
    def show_instructions(self):
        """Show instructions on how to use the application"""
        instr_window = tk.Toplevel(self)
        instr_window.title("How to Use the Forecasting Tool")
        instr_window.geometry("600x500")
        
        # Add a scrollable text area
        instr_text = scrolledtext.ScrolledText(
            instr_window,
            wrap=tk.WORD,
            padx=10,
            pady=10,
            font=("Arial", 11)
        )
        instr_text.pack(fill="both", expand=True)
        
        # Define the instructions
        instructions = """# How to Use the Doghouse Sales Forecasting Tool

    ## Step 1: Select a Model Version
    - **Dissertation Version**: More advanced model with extended features
    - Requires more data columns (Month, Net items sold, weather_score, web_traffic)
    - Uses RobustScaler and 3-layer LSTM architecture
    
    - **Company Version**: Simpler model with basic requirements
    - Needs only two columns (Month, Net items sold)
    - Uses MinMaxScaler and 2-layer LSTM

    ## Step 2: Select Your Sales CSV File
    - Click the "Browse" button to select your CSV file
    - Ensure your file has the required columns for your chosen model
    - Date format must be YYYY-MM-DD (e.g., 2023-01-01)

    ## Step 3: Validate Your CSV
    - Click the "Validate" button to check if your CSV is compatible
    - The system will verify required columns and data format
    - If validation passes, the "Run Forecast" button will be enabled

    ## Step 4: Run the Forecast
    - Click the "Run Forecast" button to generate predictions
    - The process may take a few moments to complete
    - Results will appear in the console area below

    ## Step 5: Review Results
    - The forecast plot will open automatically
    - The "final_plots" folder will open with all output files
    - Performance metrics will be displayed in the console
    - The output includes a CSV with the forecast values

    ## Tips for Best Results
    - Provide at least 24 months of historical data
    - Ensure dates are properly formatted
    - For better accuracy with the dissertation model, include all recommended columns
    """
        
        instr_text.insert(tk.END, instructions)
        instr_text.configure(state="disabled")
        
        # Add a close button
        close_btn = tk.Button(
            instr_window,
            text="Close",
            command=instr_window.destroy,
            padx=20
        )
        close_btn.pack(pady=10)

    def update_description(self, *args):
        if self.version_var.get() == "dissertation":
            self.desc_text.set(
                "Dissertation Version (Extended Features):\n"
                "• Uses RobustScaler for better handling of outliers\n"
                "• 3-layer LSTM architecture with deeper network\n"
                "• Supports advanced features (weather, web traffic)\n"
                "• Synthetic history generation for better training\n"
                "• Standardized residual calculation"
            )
        else:
            self.desc_text.set(
                "Company Version (Standard):\n"
                "• Basic forecasting pipeline\n"
                "• Uses MinMaxScaler for data normalization\n"
                "• Efficient 2-layer LSTM network\n"
                "• Simpler feature engineering\n"
                "• Additional SMAPE metric reported"
            )

    def update_requirements(self, *args):
        if self.version_var.get() == "dissertation":
            self.req_text.set(
                "CSV Requirements (Dissertation Version):\n"
                "• REQUIRED columns: 'Month', 'Net items sold', 'weather_score', 'web_traffic'\n"
                "• Optional but recommended: 'discount_pct', 'discount_flag', 'promotion'\n"
                "• Date format for Month: YYYY-MM-DD (e.g. 2023-01-01)\n"
                "• Missing optional columns will be auto-generated"
            )
        else:
            self.req_text.set(
                "CSV Requirements (Company Version):\n"
                "• Required columns: 'Month', 'Net items sold'\n"
                "• Optional: 'Discounts', 'Gross sales'\n"
                "• Date format for Month: YYYY-MM-DD (e.g. 2023-01-01)\n"
                "• At least 24 months of data recommended for best results"
            )
    
    def validate_csv(self):
        """Validate CSV file against requirements for selected model"""
        csv_path = self.file_var.get()
        if not csv_path:
            messagebox.showerror("Error", "Please select a CSV file first.")
            return
            
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            
            # Basic required columns for all versions
            basic_required_cols = ['Month', 'Net items sold']
            missing_basic = [col for col in basic_required_cols if col not in df.columns]
            
            if missing_basic:
                messagebox.showerror("Missing Basic Required Columns", 
                                    f"The CSV is missing these required columns: {', '.join(missing_basic)}")
                self.csv_validated = False
                return
                
            # Check date format
            try:
                pd.to_datetime(df['Month'])
            except:
                messagebox.showerror("Invalid Date Format",
                                    "The 'Month' column must contain valid dates (YYYY-MM-DD format)")
                self.csv_validated = False
                return
                
            # Check data length
            if len(df) < 12:
                messagebox.showwarning("Limited Data", 
                                    "CSV contains less than 12 months of data. Forecasting may be less reliable.")
            
            # Version-specific validation
            if self.version_var.get() == "dissertation":
                # For dissertation version, we need specific columns
                dissertation_required = ['weather_score', 'web_traffic']
                missing_dissertation = [col for col in dissertation_required if col not in df.columns]
                
                if missing_dissertation:
                    messagebox.showerror("Missing Dissertation Required Columns", 
                                        f"For the dissertation model, your CSV must include: {', '.join(missing_dissertation)}\n\n"
                                        f"Please switch to the company version or use a different CSV file.")
                    self._log(f"Error: This CSV is not compatible with the dissertation model.\n"
                             f"Missing required columns: {', '.join(missing_dissertation)}\n")
                    self.csv_validated = False
                    return
                
                # Check optional columns
                dissertation_optional = ['discount_pct', 'discount_flag', 'promotion', 'sin_month', 'cos_month']
                missing_optional = [col for col in dissertation_optional if col not in df.columns]
                
                if missing_optional:
                    messagebox.showinfo("Missing Optional Columns",
                                     f"These recommended columns are missing and will be auto-generated: {', '.join(missing_optional)}")
                    self._log(f"Note: Missing optional columns will be auto-generated: {', '.join(missing_optional)}\n")
            else:
                # Company version is more lenient, check optional columns
                company_optional = ['Discounts', 'Gross sales', 'discount_flag']
                missing_optional = [col for col in company_optional if col not in df.columns]
                
                if missing_optional:
                    messagebox.showinfo("Missing Optional Columns",
                                     f"These optional columns are missing and will be auto-generated: {', '.join(missing_optional)}")
                    self._log(f"Note: Missing optional columns will be auto-generated: {', '.join(missing_optional)}\n")
            
            # Mark as validated and enable run button
            self.csv_validated = True
            self.run_btn.config(state="normal")
            self.status_text.set("CSV validated successfully - Ready to run")
            
            messagebox.showinfo("Validation Successful", 
                               f"CSV file is compatible with the {self.version_var.get()} version.")
                
        except Exception as e:
            self.csv_validated = False
            messagebox.showerror("Validation Error", f"Error validating CSV: {str(e)}")
            return

    def browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV Files","*.csv")],
            title="Select sales CSV")
        if path:
            self.file_var.set(path)
            self.status_text.set(f"Selected: {os.path.basename(path)} - Please validate")

    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg)
        self.log.configure(state="disabled")
        self.log.see("end")

    def run(self):
        csv_path = self.file_var.get()
        if not csv_path:
            messagebox.showerror("Error", "Please select a CSV file first.")
            return
            
        if not self.csv_validated:
            messagebox.showerror("Validation Required", 
                               "Please validate the CSV before running the forecast.")
            return
        
        self._log(f"Running forecast on\n  {csv_path}\nPlease wait...\n")
        self._log(f"Using {self.version_var.get()} version\n")
        self.status_text.set("Processing...")
        self.update_idletasks()  # Force UI update
        
        try:
            # Dynamically run the selected predictor
            if self.version_var.get() == "dissertation":
                # The dissertation version now accepts 'log' parameter
                metrics = self.run_dissertation(csv_path)
            else:
                # The company version accepts 'log' parameter
                metrics = self.run_company(csv_path)
                
            self._log("Forecast complete!\n")
            self.status_text.set("Forecast completed successfully")
            self._open_plot(csv_path)
            self._open_output_folder()
            
        except Exception as e:
            self.status_text.set("Error occurred")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            return

    def run_dissertation(self, csv_path):
        """Run the dissertation version with log parameter"""
        try:
            import doghouse_predictor5
            return doghouse_predictor5.analyze_csv(csv_path, log=self._log)
        except Exception as e:
            self._log(f"Error in dissertation model: {str(e)}\n")
            raise

    def run_company(self, csv_path):
        """Run the company version with log parameter"""
        try:
            import doghouse_predictor2
            return doghouse_predictor2.analyze_csv(csv_path, log=self._log)
        except Exception as e:
            self._log(f"Error in company model: {str(e)}\n")
            raise

    def _open_plot(self, csv_path):
        """Open the generated forecast plot"""
        base_filename = os.path.basename(csv_path)
        plot_name = f"{base_filename}_future.png"
        plot_path = os.path.join("final_plots", plot_name)
        
        if not os.path.exists(plot_path):
            # Try with just the filename without path
            filename_only = os.path.splitext(base_filename)[0]
            plot_name = f"{filename_only}_future.png"
            plot_path = os.path.join("final_plots", plot_name)
            
        if os.path.exists(plot_path):
            if sys.platform.startswith("win"):
                os.startfile(plot_path)
            else:
                webbrowser.open(plot_path)
        else:
            self._log(f"Warning: Plot file not found at {plot_path}\n")

    def _open_output_folder(self):
        output_path = os.path.abspath("final_plots")
        if os.path.exists(output_path):
            if sys.platform.startswith("win"):
                os.startfile(output_path)
            elif sys.platform.startswith("darwin"):
                os.system(f"open '{output_path}'")
            else:  # Linux
                os.system(f"xdg-open '{output_path}'")


if __name__ == "__main__":
    app = ForecastAppToggle()
    app.after(500, app.show_instructions)
    app.mainloop()