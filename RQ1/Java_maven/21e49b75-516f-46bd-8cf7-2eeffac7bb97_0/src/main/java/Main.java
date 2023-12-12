public class Main {
    public static void main(String [] args) {
        String className = "com.example.package.YourClassName";
        try {
            Class<?> clazz = Class.forName(className);
            Package pkg = clazz.getPackage();
            String packageName = pkg.getName();
            System.out.println("Package: " + packageName);
        } catch (ClassNotFoundException e) {
            // Handle class not found exception
            e.printStackTrace();
        }
   }
}