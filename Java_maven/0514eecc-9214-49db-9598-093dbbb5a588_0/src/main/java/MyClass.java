import java.lang.reflect.Method;

public class MyClass {
    public void publicMethod1() {
        // Method implementation
    }

    public int publicMethod2(String param) {
        // Method implementation
        return 0;
    }

    private void privateMethod() {
        // Method implementation
    }

    public static void main(String[] args) {
        Class<?> myClass = MyClass.class;

        Method[] methods = myClass.getMethods();

        for (Method method : methods) {
            if (method.getDeclaringClass() == myClass) {
                System.out.println(method.getName());
            }
        }
    }
}
