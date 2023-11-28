public class Main {
    class MathOperations {
        static int add(int a, int b) { // Static method
            return a + b;
        }
    }
    public static void main(string [] args) {
        // Calling the static method without creating an object
        int result = MathOperations.add(5, 3);
    }
}