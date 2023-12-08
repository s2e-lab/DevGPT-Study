public class Main{

    public static void main(String [] args) {
        String textBlock = """
            This is a text block with variables.
            Variable 1: %s
            Variable 2: %s
            Variable 3: %s
            """;

        String variable1 = "Value 1";
        String variable2 = "Value 2";
        String variable3 = "Value 3";

        String replacedText = String.format(textBlock, variable1, variable2, variable3);
        // Alternatively, you can use textBlock.formatted(variable1, variable2, variable3);

        System.out.println(replacedText);
    }
}
