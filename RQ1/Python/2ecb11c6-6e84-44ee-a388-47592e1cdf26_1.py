# Function to collect messages
def collect_messages(_):
    prompt = inp.value
    inp.value = ''
    context.append({'role':'user', 'content':f"{prompt}"})
    response = get_completion_from_messages(context)
    context.append({'role':'assistant', 'content':f"{response}"})
    panels.append(
        pn.Row('User:', pn.pane.Markdown(prompt)))
    panels.append(
        pn.Row('Assistant:', pn.pane.Markdown(response)))
    output_div.object = pn.Column(*panels)
    return output_div  # returning Div widget
