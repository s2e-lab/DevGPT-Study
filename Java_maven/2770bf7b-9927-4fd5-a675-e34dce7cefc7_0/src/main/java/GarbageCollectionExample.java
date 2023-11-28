public class GarbageCollectionExample {
    public static void main(String[] args) {
        // Create some objects
        for (int i = 0; i < 10000; i++) {
            new GarbageCollectionExample();
        }

        // Suggest garbage collection
        System.gc();
        // Alternatively: Runtime.getRuntime().gc();
    }

    // This method is here to create some objects
    // which will later become eligible for garbage collection
    public void someMethod() {
        // Method code here
    }
}
