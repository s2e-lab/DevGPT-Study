public class Main {
    public static void main(string [] args) {
        Logger logger = LogManager.getLogger();
        logger.atInfo().withKeyValue("userId", "12345").log("User logged in");
   }
}
