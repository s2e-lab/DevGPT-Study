public class Main {
    public static void main(String [] args) {
        DependencyRequest dependencyRequest = new DependencyRequest(collectRequest, null);
        DependencyResult dependencyResult = repositorySystem.resolveDependencies(session, dependencyRequest);
   }
}
