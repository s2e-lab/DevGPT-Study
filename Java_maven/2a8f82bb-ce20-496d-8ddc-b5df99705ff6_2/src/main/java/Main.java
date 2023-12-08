public class Main {
    public static void main(string [] args) {
        @Retention(RetentionPolicy.RUNTIME)
        @Target(ElementType.METHOD)
        public @interface Observe {}
    }
}