public class Main {
    public Cliente getClienteById(int id) throws EntityNotFoundException {
        // Tenta encontrar o cliente
        // Se o cliente não for encontrado, lança a exceção
        if (!clienteRepository.existsById(id)) {
            throw new EntityNotFoundException("Cliente com ID " + id + " não encontrado.");
        }
        // Se o cliente for encontrado, o retorna
        return clienteRepository.findById(id);
    }
}