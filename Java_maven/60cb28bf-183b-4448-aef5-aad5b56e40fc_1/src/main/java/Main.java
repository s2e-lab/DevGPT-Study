public class Main {
    public static void main(string [] args) {
        File diretorioImagens = new File("caminho/do/diretorio");
        if (!diretorioImagens.exists()) {
            diretorioImagens.mkdirs();
        }
   }
}