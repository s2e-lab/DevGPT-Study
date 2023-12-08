import java.io.File;

public class Main {
    public static void main(String [] args) {
        File diretorioImagens = new File("caminho/do/diretorio");
        if (!diretorioImagens.exists()) {
            diretorioImagens.mkdirs();
        }
   }
}