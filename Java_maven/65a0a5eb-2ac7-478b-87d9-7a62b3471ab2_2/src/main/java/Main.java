public class Main {
    public static void main(String [] args) {
        RepositorySystem repositorySystem = MavenRepositorySystemUtils.newServiceLocator()
                .getRepositorySystem();
        RepositorySystemSession session = MavenRepositorySystemUtils.newSession();
   }
}