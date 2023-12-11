class NewBase:
    def method(self):
        print("NewBase method")

class Base(NewBase):
    pass

class Derived(Base):
    def method(self):
        super().method()
        print("Derived method")

d = Derived()
d.method()
