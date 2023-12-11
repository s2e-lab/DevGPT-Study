def quick_response_click(event):
    inp.value = event.obj.name  # Set the input value to the name of the button
    button_conversation.clicks += 1  # Simulate click on the chat button
    collect_messages(None)  # Automatically collect and display messages

# Define quick response buttons and their actions
quick_responses = ["Hello", "Order pizza", "Order drink", "Delivery", "End chat"]

quick_response_buttons = [pn.widgets.Button(name=response, width=150) for response in quick_responses]
for button in quick_response_buttons:
    button.on_click(quick_response_click)

# Create a row with quick response buttons
quick_response_row = pn.Row(*quick_response_buttons, css_classes=["container"], margin=(10, 0, 10, 0))

# Create the dashboard which includes the chat box, quick responses and the interactive conversation widget
dashboard = pn.Column(
    pn.panel(interactive_conversation, loading_indicator=True),
    chat_box,
    quick_response_row,
    css_classes=["container"],  # Add 'container' CSS class
)
