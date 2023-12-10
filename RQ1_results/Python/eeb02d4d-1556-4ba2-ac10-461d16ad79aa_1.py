import traceback

async def create_task(queue, operation):
    current_stack = traceback.format_stack()
    task = Task(operation, current_stack)
    await queue.put(task)
