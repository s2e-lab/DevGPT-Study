async def process_task(queue):
    while True:
        task = await queue.get()
        task.stack_trace += traceback.format_stack()
        # process task.operation here
        # ...
        print('\n'.join(task.stack_trace))  # print full stack trace if needed
        queue.task_done()
