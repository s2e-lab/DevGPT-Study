import org.junit.jupiter.api.*;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;

@Testcontainers
public class DynamicClassExecutionTest extends ContainerExecutionExtensionSupport {

    private static final String DYNAMIC_CLASS_NAME = "DynamicClass";
    private static final String DYNAMIC_CLASS_CODE =
            "public class DynamicClass {\n" +
            "    public static void main(String[] args) {\n" +
            "        System.out.println(\"Hello, dynamic class!\");\n" +
            "    }\n" +
            "}\n";

    @Container
    private static final GenericContainer<?> container = new GenericContainer<>("my-container-image:latest")
            .withClasspathResourceMapping(
                    Path.of("path/to/dynamic-class-file/"),
                    "/path/to/dynamic-class-file/",
                    BindMode.READ_ONLY);

    @BeforeEach
    public void setup() {
        container.start();
    }

    @AfterEach
    public void teardown() {
        container.stop();
    }

    @Test
    public void executeDynamicClass() throws IOException, InterruptedException {
        // Create a temporary file and write the dynamic class code to it
        File tempFile = File.createTempFile(DYNAMIC_CLASS_NAME, ".java");
        try (FileWriter writer = new FileWriter(tempFile)) {
            writer.write(DYNAMIC_CLASS_CODE);
        }

        // Compile the dynamic class file
        javax.tools.JavaCompiler compiler = javax.tools.ToolProvider.getSystemJavaCompiler();
        int compilationResult = compiler.run(null, null, null, tempFile.getAbsolutePath());
        if (compilationResult != 0) {
            // Handle compilation error
            return;
        }

        // Build a new Docker image including the compiled class
        String dockerImageName = "my-dynamic-image:latest";
        Path classFile = Path.of(tempFile.getAbsolutePath().replace(".java", ".class"));
        container.copyFileToContainer(classFile, "/path/to/dynamic-class-file/" + DYNAMIC_CLASS_NAME + ".class");
        container.execInContainer("jar", "cvf", "/path/to/dynamic-class-file/dynamic.jar",
                "-C", "/path/to/dynamic-class-file/", ".");

        // Update container image to use the newly created Docker image
        container.setDockerImageName(dockerImageName);

        // Restart the container with the updated image
        container.stop();
        container.start();

        // Execute the dynamic class inside the container
        String command = "java -cp /path/to/dynamic-class-file/dynamic.jar " + DYNAMIC_CLASS_NAME;
        String output = container.execInContainer(command).getStdout();

        // Process the output as needed
        System.out.println(output);
    }
}
