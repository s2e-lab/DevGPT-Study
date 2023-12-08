public class Main {
    public static void main(String[] args) {
        String textBlock = "This is a text block with variables.\n" +
                "Variable 1: %s\n" +
                "Variable 2: %s\n" +
                "Variable 3: %s\n";

        String variable1 = "Value 1";
        String variable2 = "Value 2";
        String variable3 = "Value 3";

        String replacedText = String.format(textBlock, variable1, variable2, variable3);

        System.out.println(replacedText);
    }
}
