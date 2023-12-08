public class Main {
    public static void main(string [] args) {
        RepositorySystem repositorySystem = MavenRepositorySystemUtils.newServiceLocator()
                .getRepositorySystem();
        RepositorySystemSession session = MavenRepositorySystemUtils.newSession();
   }
}