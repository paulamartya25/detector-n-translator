"""
main.py — Entry point alias for app.py
Run this file OR app.py, both work the same.
"""
from app import App

if __name__ == "__main__":
    app = App()
    app.mainloop()
