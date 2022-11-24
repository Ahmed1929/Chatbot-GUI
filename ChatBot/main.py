from tkinter import *
from Chat import response
# creating gui window
class GUI:
    def __init__(self):
        self.GUI_window = Tk()
        self.GUI_window.title("ChatBot")
        self.window_settings()
    def window_settings(self):
        # Window layout
        self.GUI_window.resizable(width=True, height=True)
        self.GUI_window.configure(width=500, height=500, bg="#415D99")

        # Adding icon to the window
        photo = PhotoImage(file="robotic.png")
        self.GUI_window.iconphoto(False, photo)

        # Adding text field for showing messages
        self.messages = Text(self.GUI_window, width=40, height=20, font=("Helvetica", 16), bg="#5F89E3", fg="#ffffff")
        self.messages.place(relheight=0.8, relwidth=1)
        self.messages.configure(state=DISABLED, cursor="arrow")

        # Adding scroll bar to navigate the messages
        scrollbar = Scrollbar(self.messages)
        scrollbar.place(relheight=1, relx=0.98)
        scrollbar.configure(command=self.messages.yview())

        # Adding text field for taking input from the user
        self.input = Entry(self.GUI_window, bg="#B1F5C9", fg="#000000", borderwidth=5)
        self.input.place(relwidth=0.7, relheight=0.1, relx=0.001, rely=0.8)
        self.input.focus()

        # Adding Button
        self.sendButton = Button(self.GUI_window, text="Send", fg="#000000", bg="#6740E6", command=self.chat)
        self.sendButton.place(relheight=0.1, relwidth=0.2, relx=0.800, rely=0.8)
    # Displaying the message in the text field
    def chat(self):
        message = self.input.get()
        if not message:
            return
        self.input.delete(0, END)
        bot = "Leo: " + response(message) + "\n\n"
        message = "You: " + message + "\n\n"
        self.messages.configure(state=NORMAL)
        self.messages.insert(END, message)
        self.messages.insert(END, bot)
        self.messages.configure(state=DISABLED)
if __name__ == "__main__":
    chat = GUI()
    chat.GUI_window.mainloop()
