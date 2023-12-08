public class Main {
    public static void main(string [] args) {
        try {
            Cliente cliente = getClienteById(123);
            // Faz algo com o cliente
        } catch (EntityNotFoundException e) {
            // Faz algo em resposta à exceção (por exemplo, loga o erro e/ou retorna uma mensagem de erro ao usuário)
        }
    }
}

