public class Main {
    public static void main(String [] args) {
        MDC.put("userId", "12345");
        logger.info("User logged in");

   }
}
