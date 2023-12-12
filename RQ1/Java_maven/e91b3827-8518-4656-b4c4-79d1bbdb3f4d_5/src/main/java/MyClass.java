public class MyClass {
    // Private variable
    private int privateVar = 40;

    // Private method
    private void privateMethod() {
        System.out.println("This is a private method.");
    }

    void someMethod() {
        // Accessing the private variable and method within the same class
        int value = privateVar;
        privateMethod();
    }
}
