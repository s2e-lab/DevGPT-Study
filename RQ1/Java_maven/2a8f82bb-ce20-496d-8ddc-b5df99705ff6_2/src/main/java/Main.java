public class Main {
    public static void main(String [] args) {
        @Retention(RetentionPolicy.RUNTIME)
        @Target(ElementType.METHOD)
        public @interface Observe {}
    }
}