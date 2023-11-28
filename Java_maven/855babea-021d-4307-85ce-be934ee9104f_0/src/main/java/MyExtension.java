import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.QuPathExtension;

import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.scene.layout.VBox;

public class MyExtension implements QuPathExtension {

    private QuPathGUI qupath;

    @Override
    public void installExtension(QuPathGUI qupath) {
        this.qupath = qupath;

        // Create UI
        VBox vbox = new VBox();

        TextField pathField = new TextField();
        pathField.setPromptText("Enter path");

        TextField paramField = new TextField();
        paramField.setPromptText("Enter parameters");

        Button runButton = new Button("Run");
        runButton.setOnAction(e -> run(pathField.getText(), paramField.getText()));

        Button runForProjectButton = new Button("Run for Project");
        runForProjectButton.setOnAction(e -> runForProject(pathField.getText(), paramField.getText()));

        vbox.getChildren().addAll(pathField, paramField, runButton, runForProjectButton);

        // Add to QuPath
        qupath.addWindowExtension(vbox, "My Extension");
    }

    public void run(String path, String params) {
        // Implement your "run" logic here
    }

    public void runForProject(String path, String params) {
        // Implement your "run for project" logic here
    }

    @Override
    public String getName() {
        return "My Extension";
    }

    @Override
    public String getDescription() {
        return "This is my custom extension for QuPath";
    }
}
