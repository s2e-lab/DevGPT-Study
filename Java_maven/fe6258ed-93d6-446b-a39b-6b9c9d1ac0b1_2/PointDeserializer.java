public class PointDeserializer extends JsonDeserializer<Point> {
    @Override
    public Point deserialize(JsonParser jsonParser, DeserializationContext deserializationContext) throws IOException {
        JsonNode node = jsonParser.getCodec().readTree(jsonParser);
        double x = node.get("x").asDouble();
        double y = node.get("y").asDouble();
        return new Point(x, y);
    }
}
