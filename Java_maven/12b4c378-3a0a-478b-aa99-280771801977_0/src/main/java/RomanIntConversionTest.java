import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class RomanIntConversionTest {

    @Test
    public void testConvertIntToRoman() {
        // ... other tests here ...

        // Negative tests
        assertEquals("", RomanIntConversion.convertIntToRoman(-1));
        assertEquals("", RomanIntConversion.convertIntToRoman(4000));
    }
}
