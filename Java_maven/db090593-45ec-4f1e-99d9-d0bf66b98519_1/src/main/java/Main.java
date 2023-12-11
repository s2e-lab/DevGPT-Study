import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Main {
    public static void main(String[] args) {
        List<String> strings = Arrays.asList("a", "b", "c", "d", "e");
        List<String> list = strings.stream().collect(Collectors.toList());
        String joined = strings.stream().collect(Collectors.joining(", "));
    }
}
