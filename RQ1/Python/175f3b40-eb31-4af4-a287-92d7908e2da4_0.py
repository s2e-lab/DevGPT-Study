from sqlmodel import Session, select
from typing import List

def view_todos(engine) -> List[Todo]:
    with Session(engine) as session:
        statement = select(Todo)
        todos = session.exec(statement).all()
        return todos

def add_todo(engine, title: str) -> Todo:
    with Session(engine) as session:
        new_todo = Todo(title=title, completed=False)
        session.add(new_todo)
        session.commit()
        session.refresh(new_todo)
        return new_todo

def complete_todo(engine, todo_id: int) -> Todo:
    with Session(engine) as session:
        todo = session.get(Todo, todo_id)
        if todo is not None:
            todo.completed = True
            session.add(todo)
            session.commit()
            session.refresh(todo)
        return todo
