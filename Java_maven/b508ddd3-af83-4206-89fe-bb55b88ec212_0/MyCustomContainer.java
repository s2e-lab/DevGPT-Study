import org.testcontainers.containers.GenericContainer;
import org.testcontainers.utility.MountableFile;

public class MyCustomContainer extends GenericContainer<MyCustomContainer> {

    private static final String CLASS_FILE_PATH = "/path/to/YourClass.class";

    public MyCustomContainer() {
        super("my-image:latest");
        // Configure the container, set exposed ports, environment variables, etc.
        // Mount the class file into the container
        this.withCopyFileToContainer(MountableFile.forClasspathResource("YourClass.class"), CLASS_FILE_PATH);
    }

    @Override
    protected void starting() {
        // Perform setup logic before the container starts
    }

    public void runClass() {
        // Run the added class inside the container
        String command = "java -cp " + CLASS_FILE_PATH + " YourClass";
        this.execInContainer("sh", "-c", command);
    }

    public static void main(String[] args) {
        MyCustomContainer container = new MyCustomContainer();
        container.start(); // Start the container
        container.runClass(); // Execute the added class in the container
        container.stop(); // Stop the container when done
    }
}
