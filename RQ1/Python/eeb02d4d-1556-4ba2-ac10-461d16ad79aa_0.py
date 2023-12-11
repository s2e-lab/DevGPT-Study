import asyncio
import traceback

class Task:
    def __init__(self, operation, stack_trace=None):
        self.operation = operation
        self.stack_trace = stack_trace

DEBUG_MODE = False  # Change this to True for debugging

async def create_task(queue, operation):
    current_stack = None
    if DEBUG_MODE:
        current_stack = traceback.format_stack()
    task = Task(operation, current_stack)
    await queue.put(task)

async def process_task(queue):
    while True:
        task = await queue.get()
        if DEBUG_MODE and task.stack_trace is not None:
            task.stack_trace += traceback.format_stack()
            print('\n'.join(task.stack_trace))  # print full stack trace if needed
        # process task.operation here
        # ...
        queue.task_done()

# Example usage:
queue = asyncio.Queue()
asyncio.create_task(create_task(queue, "operation1"))
asyncio.create_task(process_task(queue))
