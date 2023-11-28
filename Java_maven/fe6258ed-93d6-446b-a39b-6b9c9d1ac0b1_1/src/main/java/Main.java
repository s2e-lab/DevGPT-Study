public class Main {
    public static void main(string [] args) {
        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.addMixIn(Point.class, PointMixin.class);
   }
}
