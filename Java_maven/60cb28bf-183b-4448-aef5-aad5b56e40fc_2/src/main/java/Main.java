import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

public class Main {
    public static void main(String[] args) throws IOException {
        String diretorioImagens = "path_to_directory"; // Replace "path_to_directory" with your directory path
        String nomeImagem = "image.jpg"; // Replace "image.jpg" with your image file name
        
        File imagemOrigem = new File("path_to_image"); // Replace "path_to_image" with the source image path
        File imagemDestino = new File(diretorioImagens, nomeImagem);
        
        Files.copy(imagemOrigem.toPath(), imagemDestino.toPath());
    }
}
