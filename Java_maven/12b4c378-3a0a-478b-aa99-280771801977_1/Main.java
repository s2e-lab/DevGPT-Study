public class Main {
    public static void main(string [] args) {
        @Test
        public void testConvertIntToRomanThrowsException() {
            // Negative numbers
            assertThrows(IllegalArgumentException.class, () -> RomanIntConversion.convertIntToRoman(-1));

            // Numbers above 3999
            assertThrows(IllegalArgumentException.class, () -> RomanIntConversion.convertIntToRoman(4000));
        }
   }
}