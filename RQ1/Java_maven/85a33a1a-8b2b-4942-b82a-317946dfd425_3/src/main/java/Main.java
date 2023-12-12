public class Main {
    public static void main(String[] args) {
        String url = "example.com";

        if (shouldBlock(url)) {
            System.out.println("URL should be blocked.");
            // Drop the connection or redirect it
        } else {
            System.out.println("URL is allowed.");
            // Allow the connection
        }
    }

    private static boolean shouldBlock(String url) {
        // logic to determine if the URL should be blocked
        return url.contains("example");
    }
}
