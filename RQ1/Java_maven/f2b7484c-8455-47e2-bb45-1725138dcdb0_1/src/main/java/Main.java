@Transactional
public class Main {
    public void saveEntity(Entity entity) {
        try {
            // Repository save operation
            repository.save(entity);
        } catch (Exception ex) {
            // Log the exception or perform error handling
            throw ex; // or throw a new exception
        }
    }
}