public class BoothAlgorithm {
    public static String boothMultiply(String multiplier, String multiplicand) {
        int n = multiplier.length();
        int productLength = 2 * n;  // Length of the product

        // Initialize the product and the accumulator
        StringBuilder product = new StringBuilder();
        StringBuilder accumulator = new StringBuilder();
        for (int i = 0; i < productLength; i++) {
            product.append("0");
            accumulator.append("0");
        }

        // Extend the multiplier and multiplicand to match the product length
        multiplier = extendBinaryString(multiplier, productLength);
        multiplicand = extendBinaryString(multiplicand, productLength);

        // Perform the booth's algorithm
        for (int i = 0; i < n; i++) {
            char multiplierLSB = multiplier.charAt(n - 1);  // Least Significant Bit of the multiplier

            // Check if the last two bits of the multiplier are "01"
            if (multiplierLSB == '1' && multiplier.charAt(0) == '0') {
                addBinaryStrings(accumulator, multiplicand);  // Add multiplicand to the accumulator
            }
            // Check if the last two bits of the multiplier are "10"
            else if (multiplierLSB == '0' && multiplier.charAt(0) == '1') {
                addBinaryStrings(accumulator, negateBinaryString(multiplicand));  // Subtract multiplicand from the accumulator
            }

            // Perform arithmetic right shift on the multiplier and the accumulator
            StringBuilder sb_mult = new StringBuilder(multiplier);
            shiftRight(sb_mult);
            shiftRight(accumulator);
        }

        // Get the product from the accumulator
        product.replace(0, productLength, accumulator.toString());

        return product.toString();
    }

    // Helper method to extend a binary string to a specified length by sign extension
    private static String extendBinaryString(String binaryString, int length) {
        StringBuilder extended = new StringBuilder();
        char signBit = binaryString.charAt(0);  // Sign bit of the binary string

        for (int i = 0; i < length - binaryString.length(); i++) {
            extended.append(signBit);
        }
        extended.append(binaryString.substring(1));  // Append the original binary string (excluding the sign bit)

        return extended.toString();
    }

    // Helper method to add two binary strings and store the result in the first string
    private static void addBinaryStrings(StringBuilder str1, String str2) {
        int carry = 0;
        int n = str1.length();

        for (int i = n - 1; i >= 0; i--) {
            int sum = Character.getNumericValue(str1.charAt(i)) + Character.getNumericValue(str2.charAt(i)) + carry;
            str1.setCharAt(i, (char) ('0' + (sum % 2)));
            carry = sum / 2;
        }
    }

    // Helper method to negate a binary string (two's complement)
    private static String negateBinaryString(String binaryString) {
        StringBuilder negated = new StringBuilder();

        for (char bit : binaryString.toCharArray()) {
            negated.append(bit == '0' ? '1' : '0');
        }

        addBinaryStrings(negated, "1");  // Add 1 to the negated binary string (two's complement)

        return negated.toString();
    }

    // Helper method to perform arithmetic right shift on a binary string
    private static void shiftRight(StringBuilder binaryString) {
        int n = binaryString.length();
        char signBit = binaryString.charAt(0);  // Sign bit of the binary string

        for (int i = n - 1; i > 0; i--) {
            binaryString.setCharAt(i, binaryString.charAt(i - 1));
        }
        binaryString.setCharAt(0, signBit);  // Place the sign bit at the most significant bit
    }

    public static void main(String[] args) {
        String multiplier = "101";
        String multiplicand = "110";

        String product = boothMultiply(multiplier, multiplicand);
        System.out.println("Product: " + product);
    }
}
