public class MyClass {
    public final int constantVar = 42; // A final constant variable
    
    public final void finalMethod() { // A final method
        // Method code here
    }
}

// Extending a final class is not allowed
public class SubClass extends MyClass { // This will give an error
    // Some code here
}
