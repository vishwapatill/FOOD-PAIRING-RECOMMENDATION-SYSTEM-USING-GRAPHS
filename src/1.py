import tkinter as tk
from tkinter import ttk

class FoodRecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Food Recommendation System")

        # Main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(padx=20, pady=20)

        # Menu Section
        self.menu_frame = tk.Frame(self.root)
        self.menu_frame.pack(side=tk.TOP, fill=tk.X)

        self.search_frame = tk.Frame(self.root)
        self.search_frame.pack()

        self.home_button = tk.Button(self.menu_frame, text="Home", width=15)
        self.home_button.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.X, expand=True)

        self.set_pref_button = tk.Button(self.menu_frame, text="Set Preferences", width=15)
        self.set_pref_button.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.X, expand=True)

        self.ingr_subs_button = tk.Button(self.menu_frame, text="Ingredient Substitution", width=15)
        self.ingr_subs_button.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.X, expand=True)

        self.find_score_button = tk.Button(self.menu_frame, text="Find Score", width=15)
        self.find_score_button.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.X, expand=True)

        # Content Section
        self.content_frame = tk.Frame(self.root)
        self.content_frame.pack()

        self.home_content = tk.Label(self.content_frame, text="Home Content")
        self.home_content.pack()

        # Bind actions to buttons
        self.home_button.config(command=lambda: self.show_content("Home Content"))
        self.set_pref_button.config(command=lambda: self.show_content("Set Preferences Content"))
        self.ingr_subs_button.config(command=lambda: self.show_content("Ingredient Substitution Content"))
        self.find_score_button.config(command=lambda: self.show_content("Find Score Content"))

    def show_content(self, content):
        # Function to show different content based on button click
        self.home_content.config(text=content)

if __name__ == "__main__":
    root = tk.Tk()
    app = FoodRecommendationApp(root)
    root.mainloop()
