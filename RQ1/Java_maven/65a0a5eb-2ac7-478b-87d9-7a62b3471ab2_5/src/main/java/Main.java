public class Main {
    public static void main(String [] args) {
        DependencyNode rootNode = dependencyResult.getRoot();
        TreeDependencyVisitor visitor = new TreeDependencyVisitor(
                new TreeDependencyVisitorFactory().newInstance(session, true));
        rootNode.accept(visitor);

        visitor.getNodes().forEach(node -> {
            Dependency dependency = node.getDependency();
            System.out.println("Group ID: " + dependency.getArtifact().getGroupId());
            System.out.println("Artifact ID: " + dependency.getArtifact().getArtifactId());
            System.out.println("Version: " + dependency.getArtifact().getVersion());
            System.out.println("--------------------");
        });
   }
}