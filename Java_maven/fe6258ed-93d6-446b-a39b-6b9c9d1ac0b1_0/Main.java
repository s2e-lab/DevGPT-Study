public class Main {
    public abstract class PointMixin {
        @JsonCreator
        public PointMixin(@JsonProperty("x") double x, @JsonProperty("y") double y) {}
    }
}