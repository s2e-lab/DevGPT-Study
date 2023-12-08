import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;

public class Main {
    public static void main(String[] args) {
        ZonedDateTime utc = ZonedDateTime.now(ZoneId.of("UTC"));
        DateTimeFormatter formatter = DateTimeFormatter.ISO_INSTANT;
        String iso_timestamp = utc.format(formatter);
        System.out.println(iso_timestamp);
    }
}
