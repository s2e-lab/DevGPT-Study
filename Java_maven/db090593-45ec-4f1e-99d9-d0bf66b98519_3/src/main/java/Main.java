import java.util.Arrays;
import java.util.List;
import java.util.Optional;

public class Main {
    public static void main(String[] args) {
        List<String> strings = Arrays.asList("a", "b", "c", "d", "e");
        Optional<String> first = strings.stream().findFirst();
        Optional<String> any = strings.stream().findAny();
    }
}
