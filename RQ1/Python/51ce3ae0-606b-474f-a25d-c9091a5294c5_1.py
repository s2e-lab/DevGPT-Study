@ui.page('/')
async def main(client: Client):
    ...
    with ui.row().classes('w-full max-w-2xl mx-auto h-[500px] scroll px-2'):  # change class to "scroll"
        with ui.column().classes('w-full items-stretch'):  # here we add 'w-full'
            await chat_messages(user_id)
    ...
