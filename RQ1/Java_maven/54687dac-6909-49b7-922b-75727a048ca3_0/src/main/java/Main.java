import org.junit.jupiter.api.Test;
import org.springframework.test.context.TestPropertySource;

// ...
public class Main {
    @Test
    @TestPropertySource(properties = "OPENAI_API_KEY=test-api-key")
    public void yourTestMethod() {
    // Your test code here
    }
}
