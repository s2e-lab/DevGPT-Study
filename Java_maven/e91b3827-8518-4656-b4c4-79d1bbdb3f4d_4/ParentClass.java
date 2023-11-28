package mypackage;

public class ParentClass {
    // Protected variable
    protected int protectedVar = 30;

    // Protected method
    protected void protectedMethod() {
        System.out.println("This is a protected method.");
    }
}

package mypackage;

public class ChildClass extends ParentClass {
    void someMethod() {
        // Accessing the protected variable and method from the subclass
        int value = protectedVar;
        protectedMethod();
    }
}
