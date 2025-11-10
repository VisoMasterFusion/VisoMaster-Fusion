import tkinter as tk
from tkinter import ttk, font
import sys
import os

# This script is called before the venv is created. The Start_Portable.bat script
# sets PYTHONPATH
try:
    from app.ui.launcher.cfgtools import write_portable_cfg
except ImportError:
    # Add a fallback for running the script directly for testing or if PYTHONPATH fails
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    repo_root = os.path.abspath(os.path.join(app_dir, ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from app.ui.launcher.cfgtools import write_portable_cfg
    except ImportError:
        print(
            "Error: Could not import cfgtools. Please ensure the script is run from the correct location.",
            file=sys.stderr,
        )
        sys.exit(1)

# --- UI Styling Constants ---
BG_COLOR = "#2e3440"
TEXT_COLOR = "#d8dee9"
BUTTON_BG = "#3b4252"
BUTTON_ACTIVE_BG = "#4c566a"
BUTTON_FG = "#d8dee9"


class BranchSelectorDialog:
    """A simple Tkinter dialog to select the Git branch on first setup."""

    def __init__(self, root):
        self.root = root
        self.selected_branch = None
        self.root.title("VisoMaster Fusion - Branch Selection")
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(False, False)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TLabel",
            background=BG_COLOR,
            foreground=TEXT_COLOR,
            padding=10,
            font=("Segoe UI", 10),
        )
        style.configure("TFrame", background=BG_COLOR)
        style.configure(
            "TButton",
            background=BUTTON_BG,
            foreground=BUTTON_FG,
            borderwidth=0,
            focusthickness=0,
            padding=12,
            font=("Segoe UI", 11, "bold"),
        )
        style.map("TButton", background=[("active", BUTTON_ACTIVE_BG)])

        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        title_font = font.Font(family="Segoe UI", size=14, weight="bold")
        title_label = ttk.Label(
            main_frame,
            text="Select Installation Branch",
            font=title_font,
            anchor="center",
        )
        title_label.pack(pady=(0, 10), fill="x")

        info_label = ttk.Label(
            main_frame,
            text="Choose which version to install.\n'Main' is stable, 'Dev' has the latest features.",
            anchor="center",
            justify="center",
        )
        info_label.pack(pady=(0, 20), fill="x")

        main_button = ttk.Button(
            main_frame,
            text="📦 Main (Stable)",
            command=lambda: self.select_branch("main"),
        )
        main_button.pack(fill="x", ipady=5)

        dev_button = ttk.Button(
            main_frame, text="🛠️ Dev (Latest)", command=lambda: self.select_branch("dev")
        )
        dev_button.pack(fill="x", ipady=5, pady=(10, 0))

        self.center_window()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def select_branch(self, branch_name):
        self.selected_branch = branch_name
        print(f"Branch selected: {self.selected_branch}")
        self.root.destroy()

    def on_closing(self):
        print("Branch selection cancelled by user.")
        self.selected_branch = None
        self.root.destroy()

    def center_window(self):
        self.root.update_idletasks()
        width = 450
        height = 250
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def run(self):
        self.root.mainloop()
        return self.selected_branch


def main():
    root = tk.Tk()
    dialog = BranchSelectorDialog(root)
    chosen_branch = dialog.run()

    if chosen_branch:
        write_portable_cfg({"BRANCH": chosen_branch})
        sys.exit(0)  # Success
    else:
        # User closed the window without selecting
        sys.exit(1)  # Failure/Cancel


if __name__ == "__main__":
    main()
