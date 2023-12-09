import java.io.File;

public class Main {
    public static void main(String[] args) {
        File diretorioImagens = new File("path_to_directory"); // Replace "path_to_directory" with your directory path
        String nomeImagem = "image.jpg"; // Replace "image.jpg" with your image file name
        
        String caminhoImagem = diretorioImagens.getAbsolutePath() + "/" + nomeImagem;
        // Execute the SQL statement to insert the path into the database
    }
}
