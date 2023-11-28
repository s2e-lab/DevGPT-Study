public class Main {
    public static void main(string [] args) {
        Artifact artifact = new DefaultArtifact("your.groupId", "your.artifactId", "your.version");
        Dependency dependency = new Dependency(artifact, "compile");

        CollectRequest collectRequest = new CollectRequest();
        collectRequest.setRoot(dependency);
   }
}