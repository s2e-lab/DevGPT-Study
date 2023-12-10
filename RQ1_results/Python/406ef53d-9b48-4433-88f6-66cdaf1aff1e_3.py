window = tk.Tk()
text_box = tk.Text(window)
text_box.pack()

def on_button_click():
    post = get_post()
    text_box.insert('end', json.dumps(post, indent=2))

button = tk.Button(window, text="Get post", command=on_button_click)
button.pack()

window.mainloop()
