public class Main {
    public static void main(string [] args) {
        File imagemDestino = new File(diretorioImagens, nomeImagem);
        Files.copy(imagemOrigem.toPath(), imagemDestino.toPath());
   }
}
