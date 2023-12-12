import com.fasterxml.jackson.databind.ObjectMapper;

public class Main {
    public static void main(String [] args) {
        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.addMixIn(Point.class, PointMixin.class);
   }
}
