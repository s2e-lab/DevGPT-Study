import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class RomanIntConversionTest {

    @Test
    public void testConvertIntToRoman() {
        // ... other tests here ...

        // Negative tests
        assertThrows(IllegalArgumentException.class, () -> RomanIntConversion.convertIntToRoman(-1));
        assertThrows(IllegalArgumentException.class, () -> RomanIntConversion.convertIntToRoman(4000));
    }
}
