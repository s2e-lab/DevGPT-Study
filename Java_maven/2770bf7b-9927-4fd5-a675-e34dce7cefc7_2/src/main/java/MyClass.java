package mypackage;

public class MyClass {
    protected int protectedVar;
    
    protected void protectedMethod() {
        // Method code here
    }
}

package mypackage.subpackage;

public class SubClass extends MyClass {
    void someMethod() {
        protectedVar = 42; // Can access protectedVar from the superclass
        protectedMethod(); // Can access protectedMethod from the superclass
    }
}
