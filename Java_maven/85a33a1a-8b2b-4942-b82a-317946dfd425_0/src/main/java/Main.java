public class Main {
    private boolean shouldBlock(String url) {
        // Send URL to LLM
        // LLM checks the URL and possibly other features
        // If LLM identifies it as ad-related, it returns true, else false

        return url.contains("ad");
    }
}