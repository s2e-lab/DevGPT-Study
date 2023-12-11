class RecursiveClass:
    
    def countdown(self, n):
        if n <= 0:
            print("Done!")
            return
        print(n)
        self.countdown(n-1)  # Recursive call using self

r = RecursiveClass()
r.countdown(3)
