import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class Main {
    @Test
    public void testConvertIntToRomanThrowsException() {
        // Negative numbers
        assertThrows(IllegalArgumentException.class, () -> RomanIntConversion.convertIntToRoman(-1));

        // Numbers above 3999
        assertThrows(IllegalArgumentException.class, () -> RomanIntConversion.convertIntToRoman(4000));
    }
}