dashboard = pn.Column(
    interactive_conversation,  # No need for pn.panel wrapper here
    inp,
    pn.Row(button_conversation),
)
