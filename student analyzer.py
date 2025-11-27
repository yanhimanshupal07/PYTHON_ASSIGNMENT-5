import os
import sys
import json
from pathlib import Path
import logging
from typing import List, Dict, Tuple
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
# --------------------
# Setup logging
# --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
 
# --------------------
# Constants / Paths
# --------------------
ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
 
SAMPLE_CSV = DATA_DIR / "sample_student_scores.csv"
CLEANED_CSV = OUTPUT_DIR / "cleaned_student_data.csv"
SUMMARY_CSV = OUTPUT_DIR / "student_summary.csv"
DASHBOARD_PNG = OUTPUT_DIR / "student_performance_dashboard.png"
SUMMARY_TXT = OUTPUT_DIR / "performance_summary.txt"
 
# --------------------
# Utility functions
# --------------------
def safe_read_csv(path: Path) -> pd.DataFrame:
   """Read CSV with basic exception handling and return DataFrame."""
   try:
       df = pd.read_csv(path)
       logging.info(f"Loaded data from {path}")
       return df
   except FileNotFoundError:
       logging.error(f"File not found: {path}")
       raise
   except pd.errors.EmptyDataError:
       logging.error(f"No data: {path}")
       raise
   except Exception as e:
       logging.error(f"Error reading {path}: {e}")
       raise
 
def ensure_sample_data():
   """Create a small sample CSV if none exists (helps when grading/demo)."""
   if SAMPLE_CSV.exists():
       return
   sample = pd.DataFrame([
       {"Name":"Aman Kumar","Roll_No":"23BCA001","Gender":"M","Subject":"Math","Marks":78,"Attendance":92,"Semester":1},
       {"Name":"Aman Kumar","Roll_No":"23BCA001","Gender":"M","Subject":"Physics","Marks":72,"Attendance":92,"Semester":1},
       {"Name":"Aman Kumar","Roll_No":"23BCA001","Gender":"M","Subject":"Chemistry","Marks":81,"Attendance":92,"Semester":1},
       {"Name":"Nisha Sharma","Roll_No":"23BCA002","Gender":"F","Subject":"Math","Marks":88,"Attendance":95,"Semester":1},
       {"Name":"Nisha Sharma","Roll_No":"23BCA002","Gender":"F","Subject":"Physics","Marks":91,"Attendance":95,"Semester":1},
       {"Name":"Nisha Sharma","Roll_No":"23BCA002","Gender":"F","Subject":"Chemistry","Marks":85,"Attendance":95,"Semester":1},
       {"Name":"Ravi Verma","Roll_No":"23BCA003","Gender":"M","Subject":"Math","Marks":54,"Attendance":68,"Semester":1},
       {"Name":"Ravi Verma","Roll_No":"23BCA003","Gender":"M","Subject":"Physics","Marks":47,"Attendance":68,"Semester":1},
       {"Name":"Ravi Verma","Roll_No":"23BCA003","Gender":"M","Subject":"Chemistry","Marks":50,"Attendance":68,"Semester":1},
       {"Name":"Priya Singh","Roll_No":"23BCA004","Gender":"F","Subject":"Math","Marks":96,"Attendance":98,"Semester":1},
       {"Name":"Priya Singh","Roll_No":"23BCA004","Gender":"F","Subject":"Physics","Marks":94,"Attendance":98,"Semester":1},
       {"Name":"Priya Singh","Roll_No":"23BCA004","Gender":"F","Subject":"Chemistry","Marks":97,"Attendance":98,"Semester":1},
   ])
   sample.to_csv(SAMPLE_CSV, index=False)
   logging.info(f"Sample dataset created at {SAMPLE_CSV}")
 
# --------------------
# OOP Modeling
# --------------------
class Student:
   """Student model storing marks per subject and metadata."""
   def __init__(self, name: str, roll_no: str, gender: str = None):
       self.name = name
       self.roll_no = roll_no
       self.gender = gender
       self.marks: Dict[str, float] = {}  # subject -> marks
 
   def add_mark(self, subject: str, marks: float):
       self.marks[subject] = float(marks)
 
   def total(self) -> float:
       return sum(self.marks.values()) if self.marks else 0.0
 
   def average(self) -> float:
       return self.total() / len(self.marks) if self.marks else 0.0
 
   def grade(self) -> str:
       avg = self.average()
       if avg >= 90: return "A+"
       if avg >= 80: return "A"
       if avg >= 70: return "B"
       if avg >= 60: return "C"
       if avg >= 50: return "D"
       return "F"
 
   def to_dict(self) -> dict:
       d = {
           "Name": self.name,
           "Roll_No": self.roll_no,
           "Gender": self.gender,
           "Total": self.total(),
           "Average": round(self.average(),2),
           "Grade": self.grade()
       }
       d.update({f"Mark_{sub}": mark for sub, mark in self.marks.items()})
       return d
 
   def __str__(self):
       return f"{self.roll_no} - {self.name} | Avg: {self.average():.2f} | Grade: {self.grade()}"
 
# --------------------
# Manager
# --------------------
class StudentManager:
   def __init__(self):
       self.students: Dict[str, Student] = {}  # roll_no -> Student
       self.df: pd.DataFrame = pd.DataFrame()
 
   def load_csv(self, path: Path):
       try:
           df = safe_read_csv(path)
       except Exception:
           raise
       # Basic validations & cleaning
       required_cols = {"Name","Roll_No","Subject","Marks"}
       if not required_cols.issubset(set(df.columns)):
           raise ValueError(f"CSV must contain columns: {required_cols}")
 
       # Fill attendance and other optional columns if missing
       if "Attendance" not in df.columns:
           df["Attendance"] = np.nan
 
       # Convert Marks to numeric, coerce errors to NaN then drop rows
       df["Marks"] = pd.to_numeric(df["Marks"], errors="coerce")
       df = df.dropna(subset=["Marks","Name","Roll_No","Subject"])
       # Ensure marks in 0-100
       df = df[(df["Marks"] >= 0) & (df["Marks"] <= 100)]
       # store cleaned df
       self.df = df.copy()
       logging.info("CSV cleaned and loaded into manager.")
 
   def build_students(self):
       if self.df.empty:
           logging.error("No loaded DataFrame to build students from.")
           return
       grouped = self.df.groupby(["Roll_No","Name"])
       for (roll, name), g in grouped:
           gender = g.get("Gender", pd.Series([None])).iloc[0] if "Gender" in g else None
           student = Student(name=name, roll_no=roll, gender=gender)
           for _, row in g.iterrows():
               student.add_mark(row["Subject"], row["Marks"])
           self.students[roll] = student
       logging.info(f"Built {len(self.students)} Student objects.")
 
   def student_summary_df(self) -> pd.DataFrame:
       rows = [s.to_dict() for s in self.students.values()]
       df_summary = pd.DataFrame(rows)
       # reorder columns
       cols = ["Roll_No","Name","Gender"] + [c for c in df_summary.columns if c.startswith("Mark_")] + ["Total","Average","Grade"]
       cols = [c for c in cols if c in df_summary.columns]
       return df_summary[cols]
 
   def top_bottom_performers(self, top_n=3) -> Tuple[List[Student], List[Student]]:
       students_sorted = sorted(self.students.values(), key=lambda s: s.average(), reverse=True)
       top = students_sorted[:top_n]
       bottom = students_sorted[-top_n:] if len(students_sorted) >= top_n else students_sorted[::-1]
       return top, bottom
 
   def subject_wise_stats(self) -> pd.DataFrame:
       # subject level mean, min, max
       if self.df.empty:
           return pd.DataFrame()
       stats = self.df.groupby("Subject")["Marks"].agg(["mean","min","max","std"]).reset_index()
       stats = stats.rename(columns={"mean":"Mean","min":"Min","max":"Max","std":"StdDev"})
       return stats
 
# --------------------
# Visualization
# --------------------
def create_dashboard(manager: StudentManager, out_path: Path = DASHBOARD_PNG):
   """Create 2x2 subplot dashboard:
     1) Bar: Average marks by student
     2) Pie: Grade distribution
     3) Line: Subject-wise average (trend across subjects)
     4) Scatter: Attendance vs Average marks (if attendance present)
   """
   df_summary = manager.student_summary_df()
   if df_summary.empty:
       logging.error("No summary data to plot.")
       return
 
   # Bar chart: Average by student
   x = df_summary["Name"]
   y = df_summary["Average"]
 
   # Pie: grade distribution
   grade_counts = df_summary["Grade"].value_counts()
 
   # Subject-wise average
   subj_stats = manager.subject_wise_stats()
   subj_mean = subj_stats[["Subject","Mean"]] if not subj_stats.empty else None
 
   # Attendance vs Average if Attendance exists in original df
   attendance_exists = "Attendance" in manager.df.columns
   attendance_df = None
   if attendance_exists:
       # compute attendance by Roll_No (mean)
       attendance_df = manager.df.groupby("Roll_No")["Attendance"].mean().reset_index()
       # join to summary
       att_join = df_summary.merge(attendance_df, on="Roll_No", how="left")
   else:
       att_join = df_summary.copy()
       att_join["Attendance"] = np.nan
 
   # Create figure
   fig, axes = plt.subplots(2,2, figsize=(14,10))
   plt.subplots_adjust(hspace=0.4, wspace=0.3)
 
   # 1 Bar chart
   axes[0,0].bar(x, y)
   axes[0,0].set_title("Average Marks by Student")
   axes[0,0].set_ylabel("Average Marks")
   axes[0,0].tick_params(axis='x', rotation=45)
 
   # 2 Pie chart
   axes[0,1].pie(grade_counts.values, labels=grade_counts.index, autopct="%1.1f%%", startangle=90)
   axes[0,1].set_title("Grade Distribution")
 
   # 3 Line chart (subjects)
   if subj_mean is not None:
       axes[1,0].plot(subj_mean["Subject"], subj_mean["Mean"], marker='o')
       axes[1,0].set_title("Subject-wise Average Marks")
       axes[1,0].set_ylabel("Average Marks")
       axes[1,0].tick_params(axis='x', rotation=45)
   else:
       axes[1,0].text(0.5,0.5,"No subject stats available", ha='center')
 
   # 4 Scatter Attendance vs Average
   axes[1,1].scatter(att_join["Attendance"], att_join["Average"])
   axes[1,1].set_title("Attendance vs Average Marks")
   axes[1,1].set_xlabel("Attendance (%)")
   axes[1,1].set_ylabel("Average Marks")
 
   fig.suptitle("Student Performance Dashboard", fontsize=16)
   plt.tight_layout(rect=[0,0,1,0.96])
   fig.savefig(out_path)
   logging.info(f"Dashboard saved to {out_path}")
 
# --------------------
# Reporting & Export
# --------------------
def export_outputs(manager: StudentManager):
   # cleaned df
   manager.df.to_csv(CLEANED_CSV, index=False)
   logging.info(f"Cleaned data exported to {CLEANED_CSV}")
 
   # summary csv
   summary_df = manager.student_summary_df()
   summary_df.to_csv(SUMMARY_CSV, index=False)
   logging.info(f"Student summary exported to {SUMMARY_CSV}")
 
   # textual summary
   top, bottom = manager.top_bottom_performers(top_n=3)
   total_consumption = summary_df["Total"].sum() if "Total" in summary_df.columns else 0
   class_avg = summary_df["Average"].mean() if "Average" in summary_df.columns else 0
 
   with open(SUMMARY_TXT, "w") as f:
       f.write("Performance Summary Report\n")
       f.write("=========================\n")
       f.write(f"Total students: {len(manager.students)}\n")
       f.write(f"Class Average (Avg of student averages): {class_avg:.2f}\n")
       f.write("\nTop performers:\n")
       for s in top:
           f.write(f"- {s.roll_no} | {s.name} : Avg {s.average():.2f}\n")
       f.write("\nBottom performers:\n")
       for s in bottom:
           f.write(f"- {s.roll_no} | {s.name} : Avg {s.average():.2f}\n")
   logging.info(f"Text summary exported to {SUMMARY_TXT}")
 
# --------------------
# CLI
# --------------------
def run_cli():
   ensure_sample_data()
   manager = StudentManager()
   print("Smart Student Performance Analyzer (Capstone)\n")
   print(f"Sample CSV (if none provided) is at {SAMPLE_CSV}\n")
 
   while True:
       print("\nMenu:")
       print("1. Load CSV dataset")
       print("2. Preview cleaned data")
       print("3. Build student objects & compute stats")
       print("4. Show student summary table")
       print("5. Create dashboard and save")
       print("6. Export outputs (CSV + summary)")
       print("7. Quick run (load sample -> build -> export -> dashboard)")
       print("0. Exit")
       choice = input("Enter choice: ").strip()
 
       try:
           if choice == "1":
               path_input = input(f"Enter CSV path (default: {SAMPLE_CSV}): ").strip()
               if path_input == "":
                   path = SAMPLE_CSV
               else:
                   path = Path(path_input)
               manager.load_csv(Path(path))
               print("Dataset loaded and cleaned.")
           elif choice == "2":
               if manager.df.empty:
                   print("No data loaded. Use option 1.")
               else:
                   print(manager.df.head(10).to_string(index=False))
           elif choice == "3":
               if manager.df.empty:
                   print("Load data first.")
               else:
                   manager.build_students()
                   print(f"Built {len(manager.students)} students.")
           elif choice == "4":
               if not manager.students:
                   print("Build students first (option 3).")
               else:
                   print(manager.student_summary_df().to_string(index=False))
           elif choice == "5":
               if not manager.students:
                   print("Build students first (option 3).")
               else:
                   create_dashboard(manager)
                   print(f"Dashboard saved to {DASHBOARD_PNG}")
           elif choice == "6":
               if manager.df.empty:
                   print("No data loaded.")
               else:
                   export_outputs(manager)
                   print("Outputs exported to output/ folder.")
           elif choice == "7":
               # quick run with sample
               manager.load_csv(SAMPLE_CSV)
               manager.build_students()
               export_outputs(manager)
               create_dashboard(manager)
               print("Quick run completed. Check output/ folder.")
           elif choice == "0":
               print("Goodbye.")
               break
           else:
               print("Invalid choice.")
       except Exception as e:
           logging.exception("An error occurred during operation.")
 
if __name__ == "__main__":
   run_cli()